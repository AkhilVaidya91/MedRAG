CREATE OR REPLACE FUNCTION match_images (
  query_embedding vector(1152),
  match_count int
)
RETURNS TABLE (
  id int8,
  similarity float
)
LANGUAGE sql
AS $$
  SELECT 
    id,
    (image_embedding <#> query_embedding) as similarity
  FROM radiology_report
  WHERE image_embedding IS NOT NULL
  ORDER BY image_embedding <#> query_embedding ASC
  LIMIT LEAST(match_count, 200);
$$;


CREATE OR REPLACE FUNCTION match_texts (
  query_embedding vector(1152),
  match_count int
)
RETURNS TABLE (
  id int8,
  similarity float
)
LANGUAGE sql
AS $$
  SELECT 
    id,
    (text_embedding <#> query_embedding) as similarity
  FROM radiology_report
  WHERE text_embedding IS NOT NULL
  ORDER BY text_embedding <#> query_embedding ASC
  LIMIT LEAST(match_count, 200);
$$;