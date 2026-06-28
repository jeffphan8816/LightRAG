"""SQL templates for PostgreSQL storage backends.

Read-path templates for queries, plus DDL templates for table creation
during initialization (CREATE TABLE IF NOT EXISTS).
"""

DDL_TEMPLATES = {
    "LIGHTRAG_VDB_CHUNKS": """CREATE TABLE LIGHTRAG_VDB_CHUNKS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    full_doc_id VARCHAR(256),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content TEXT,
                    content_vector VECTOR(dimension),
                    file_path TEXT NULL,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_VDB_CHUNKS_PK PRIMARY KEY (workspace, id)
                    )""",
    "LIGHTRAG_VDB_ENTITY": """CREATE TABLE LIGHTRAG_VDB_ENTITY (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_name VARCHAR(512),
                    content TEXT,
                    content_vector VECTOR(dimension),
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids VARCHAR(255)[] NULL,
                    file_path TEXT NULL,
                    CONSTRAINT LIGHTRAG_VDB_ENTITY_PK PRIMARY KEY (workspace, id)
                    )""",
    "LIGHTRAG_VDB_RELATION": """CREATE TABLE LIGHTRAG_VDB_RELATION (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    source_id VARCHAR(512),
                    target_id VARCHAR(512),
                    content TEXT,
                    content_vector VECTOR(dimension),
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids VARCHAR(255)[] NULL,
                    file_path TEXT NULL,
                    CONSTRAINT LIGHTRAG_VDB_RELATION_PK PRIMARY KEY (workspace, id)
                    )""",
    "LIGHTRAG_LLM_CACHE": """CREATE TABLE LIGHTRAG_LLM_CACHE (
                    workspace varchar(255) NOT NULL,
                    id varchar(255) NOT NULL,
                    original_prompt TEXT,
                    return_value TEXT,
                    chunk_id VARCHAR(255) NULL,
                    cache_type VARCHAR(32),
                    queryparam JSONB NULL,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_LLM_CACHE_PK PRIMARY KEY (workspace, id)
                    )""",
}

SQL_TEMPLATES = {
    "get_by_id_full_docs": """SELECT id, COALESCE(content, '') as content,
                                COALESCE(doc_name, '') as file_path,
                                sidecar_location,
                                parse_format,
                                content_hash,
                                process_options,
                                COALESCE(chunk_options, '{}'::jsonb) as chunk_options,
                                parse_engine
                                FROM LIGHTRAG_DOC_FULL WHERE workspace=$1 AND id=$2
                            """,
    "get_by_id_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                chunk_order_index, full_doc_id, file_path,
                                COALESCE(llm_cache_list, '[]'::jsonb) as llm_cache_list,
                                COALESCE(heading, '{}'::jsonb) as heading,
                                COALESCE(sidecar, '{}'::jsonb) as sidecar,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1 AND id=$2
                            """,
    "get_by_id_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1 AND id=$2
                               """,
    "get_by_ids_full_docs": """SELECT id, COALESCE(content, '') as content,
                                 COALESCE(doc_name, '') as file_path,
                                 sidecar_location,
                                 parse_format,
                                 content_hash,
                                 process_options,
                                 COALESCE(chunk_options, '{}'::jsonb) as chunk_options,
                                 parse_engine
                                 FROM LIGHTRAG_DOC_FULL WHERE workspace=$1 AND id = ANY($2)
                            """,
    "get_by_ids_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                  chunk_order_index, full_doc_id, file_path,
                                  COALESCE(llm_cache_list, '[]'::jsonb) as llm_cache_list,
                                  COALESCE(heading, '{}'::jsonb) as heading,
                                  COALESCE(sidecar, '{}'::jsonb) as sidecar,
                                  EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                  EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                   FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1 AND id = ANY($2)
                                """,
    "get_by_ids_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1 AND id = ANY($2)
                                """,
    "filter_keys": "SELECT id FROM {table_name} WHERE workspace=$1 AND id IN ({ids})",
    "upsert_llm_response_cache": """INSERT INTO LIGHTRAG_LLM_CACHE(workspace,id,original_prompt,return_value,chunk_id,cache_type,queryparam)
                                      VALUES ($1, $2, $3, $4, $5, $6, $7)
                                      ON CONFLICT (workspace,id) DO UPDATE
                                      SET original_prompt = EXCLUDED.original_prompt,
                                      return_value=EXCLUDED.return_value,
                                      chunk_id=EXCLUDED.chunk_id,
                                      cache_type=EXCLUDED.cache_type,
                                      queryparam=EXCLUDED.queryparam,
                                      update_time = CURRENT_TIMESTAMP
                                     """,
    "relationships": """
                     SELECT source_id AS src_id,
                            target_id AS tgt_id,
                            EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at
                     FROM {table_name}
                     WHERE workspace = $1
                       AND content_vector <=> $4::{vector_cast} < $2
                     ORDER BY content_vector <=> $4::{vector_cast}
                     LIMIT $3;
                     """,
    "entities": """
                SELECT entity_name,
                       EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at
                FROM {table_name}
                WHERE workspace = $1
                  AND content_vector <=> $4::{vector_cast} < $2
                ORDER BY content_vector <=> $4::{vector_cast}
                LIMIT $3;
                """,
    "chunks": """
              SELECT id,
                     content,
                     file_path,
                     EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at
              FROM {table_name}
              WHERE workspace = $1
                AND content_vector <=> $4::{vector_cast} < $2
              ORDER BY content_vector <=> $4::{vector_cast}
              LIMIT $3;
              """,
}
