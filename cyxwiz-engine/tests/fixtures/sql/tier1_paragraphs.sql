-- base_corpus_v3, Tier 1 practical teaching books (owner D1, 2026-10-11).
-- Input: tier1_books.zip through DataInput "ZIP text document" (one row per
-- member: member_path, member_sha256, text; bytes identical to the raw downloads).
-- Output: one row per cleaned paragraph, in reading order. No split, no admission.
-- Cleaning:
--  * Gutenberg (Baxter Parts 1-4, Ryle): body between START/END markers; a
--    leading Transcriber's Note (up to the first "PART n." / title heading) and
--    a trailing Transcriber's Note are removed.
--  * CCEL: Watson ends before "Index of Scripture References"; Law ends before
--    "Appendix A" (the later Methuen/Everyman introductions are excluded).
--  * Paragraphs = blank-line blocks. Dropped: separator rules, publisher and
--    catalogue metadata (ISBN, addresses, web links, CCEL header fields),
--    contents pages, brace diagrams, letter-spaced "T H E E N D", footnote
--    headings and footnote bodies ("[n] ..." in Gutenberg and Law; Watson's
--    "[n]" are list numbers and are kept), page/printer lines, CCEL catalogue header.
--  * Inline footnote markers [n] / [a] removed (Gutenberg, Law); _italic_ and
--    =bold= markup removed; whitespace collapsed.
WITH meta(source_id, title, author, work_family, role, format) AS (VALUES
 ('pg41633', 'A Christian Directory, Part 1: Christian Ethics', 'Richard Baxter', 'baxter_christian_directory', 'practical_christian_ethics', 'gutenberg'),
 ('pg43800', 'A Christian Directory, Part 2: Christian Economics', 'Richard Baxter', 'baxter_christian_directory', 'practical_christian_ethics', 'gutenberg'),
 ('pg44655', 'A Christian Directory, Part 3: Christian Ecclesiastics', 'Richard Baxter', 'baxter_christian_directory', 'practical_christian_ethics', 'gutenberg'),
 ('pg43967', 'A Christian Directory, Part 4: Christian Politics', 'Richard Baxter', 'baxter_christian_directory', 'practical_christian_ethics', 'gutenberg'),
 ('pg38162', 'Practical Religion', 'J. C. Ryle', 'ryle_practical_religion', 'practical_christian_ethics', 'gutenberg'),
 ('ccel_watson_commandments', 'The Ten Commandments', 'Thomas Watson', 'watson_ten_commandments', 'practical_christian_ethics', 'ccel_watson'),
 ('ccel_law_serious_call', 'A Serious Call to a Devout and Holy Life', 'William Law', 'law_serious_call', 'practical_christian_ethics', 'ccel_law')
), src AS (
 SELECT m.*, i.member_path, i.member_sha256, replace(i.text, chr(13), '') AS t
 FROM input i JOIN meta m ON m.source_id = split_part(i.member_path, '/', 1)
), body AS (
 SELECT * EXCLUDE (t),
  CASE format
   WHEN 'gutenberg' THEN regexp_extract(t, '(?s)\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\n(.*?)\*\*\* END OF THE PROJECT GUTENBERG EBOOK', 1)
   WHEN 'ccel_watson' THEN regexp_extract(t, '(?s)^\s*_{10,}\n.*?\n\s*_{10,}\n(.*?)\nIndex of Scripture References', 1)
   WHEN 'ccel_law' THEN regexp_extract(t, '(?s)^\s*_{10,}\n.*?\n\s*_{10,}\n(.*?)\n\s*Appendix A\s*\n', 1)
  END AS b
 FROM src
), lead_note AS (
 SELECT * EXCLUDE (b),
  CASE WHEN format = 'gutenberg' AND strpos(b, 'Transcriber''s') BETWEEN 1 AND 3000
       THEN regexp_replace(b, '(?s)Transcriber''s [Nn]otes?:\n.*?\n(PART [IVX]+\.|PRACTICAL RELIGION)\n', '\1' || chr(10))
       ELSE b END AS b
 FROM body
), trail_note AS (
 SELECT * EXCLUDE (b),
  CASE WHEN format = 'gutenberg' AND strpos(b, chr(10) || 'Transcriber''s') > length(b) / 2
       THEN regexp_replace(b, '(?s)\nTranscriber''s [Nn]otes?:.*$', '')
       ELSE b END AS b
 FROM lead_note
), checked AS (
 SELECT *, CASE WHEN length(trim(b)) > 100000 AND strpos(b, 'Transcriber''s') = 0 THEN b
                ELSE error('Body boundary or transcriber note not resolved for ' || source_id) END AS body
 FROM trail_note
), parts AS (
 SELECT c.* EXCLUDE (b, body), p.raw, p.paragraph_index
 FROM checked c, unnest(regexp_split_to_array(c.body, '\n[ \t]*\n+')) WITH ORDINALITY AS p(raw, paragraph_index)
), stripped AS (
 SELECT *, trim(regexp_replace(raw, '(?m)^\s*_{10,}\s*$', '', 'g')) AS p FROM parts
), kept AS (
 SELECT * FROM stripped
 WHERE p <> ''
   AND NOT regexp_matches(p, '(ISBN|P\.O\. Box|FAX|www\.|https?:|Produced by|Distributed Proofreading|Print Basis:|CCEL Subjects|LC Call no|LC Subjects|Creator\(s\):|Rights: Public Domain)')
   AND NOT regexp_full_match(p, '(?is)\s*(footnotes|(table of )?contents)\.?:?\s*')
   AND NOT (format IN ('gutenberg', 'ccel_law') AND regexp_matches(p, '^\[[0-9a-z]+\]\s'))
   AND NOT (len(list_filter(string_split(p, chr(10)), x -> regexp_matches(x, '\s{2,}([0-9]+|[ivxlcdm]+|Page)\s*$'))) * 2
            >= len(string_split(p, chr(10))))
   AND len(p) - len(replace(p, '{', '')) < 3
), cleaned AS (
 SELECT *,
  trim(regexp_replace(
    regexp_replace(regexp_replace(
      CASE WHEN format IN ('gutenberg', 'ccel_law') THEN regexp_replace(p, '\s?\[[0-9a-z]{1,3}\]', '', 'g') ELSE p END,
      '_([^_\n]+)_', '\1', 'g'), '=([^=\n]+)=', '\1', 'g'),
    '\s+', ' ', 'g')) AS text
 FROM kept
)
SELECT source_id, source_id || ':p' || lpad(CAST(paragraph_index AS VARCHAR), 5, '0') AS paragraph_id,
       CAST(paragraph_index AS BIGINT) AS paragraph_index, title, author, work_family, role,
       member_path, member_sha256, text,
       CAST(len(string_split(text, ' ')) AS BIGINT) AS word_count,
       false AS training_admitted, 'unassigned' AS split
FROM cleaned
WHERE text <> ''
  AND NOT regexp_full_match(text, '([A-Z] ){2,}[A-Z]|Page|Indexes|by|[* ]+|END OF VOL\. I\.|JOHN CHILDS AND SON, BUNGAY\.')
ORDER BY source_id, paragraph_index
