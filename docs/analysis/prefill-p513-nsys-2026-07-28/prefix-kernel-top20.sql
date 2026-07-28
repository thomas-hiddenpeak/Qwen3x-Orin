WITH named_nvtx AS (
  SELECT
    n.start,
    n.end,
    COALESCE(NULLIF(n.text, ''), text_string.value) AS nvtx_name
  FROM NVTX_EVENTS AS n
  LEFT JOIN StringIds AS text_string ON text_string.id = n.textId
),
prefix_ranges AS (
  SELECT start, end
  FROM named_nvtx
  WHERE nvtx_name = 'q3x.prefill.prefix_tile'
    AND end IS NOT NULL
),
prefix_kernels AS (
  SELECT k.*
  FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
  WHERE EXISTS (
    SELECT 1
    FROM prefix_ranges AS p
    WHERE k.start >= p.start
      AND k.end <= p.end
  )
)
SELECT
  short_string.value AS short_name,
  demangled_string.value AS demangled_name,
  COUNT(*) AS call_count,
  SUM(k.end - k.start) AS total_time_ns,
  ROUND(SUM(k.end - k.start) / 1000000.0, 6) AS total_time_ms,
  ROUND(AVG(k.end - k.start) / 1000.0, 3) AS average_time_us,
  ROUND(MAX(k.end - k.start) / 1000.0, 3) AS max_time_us
FROM prefix_kernels AS k
JOIN StringIds AS short_string ON short_string.id = k.shortName
JOIN StringIds AS demangled_string ON demangled_string.id = k.demangledName
GROUP BY
  k.shortName,
  k.demangledName,
  short_string.value,
  demangled_string.value
ORDER BY total_time_ns DESC
LIMIT 20;
