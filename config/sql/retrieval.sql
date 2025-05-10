SELECT id, chunk
FROM t_docs_chunks
ORDER BY embedding <#> :embedding::vector
LIMIT :top_k;