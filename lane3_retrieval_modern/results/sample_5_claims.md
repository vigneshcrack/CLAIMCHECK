# Lane 3 — sanity check on Lane 2's 5 sample claims (top-5)

SBERT `sentence-transformers/all-mpnet-base-v2` + FAISS; rerank = cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` over the SBERT top-20.

## Side by side: rank of the gold evidence

| Claim ID | Claim | Label | BM25 (Lane 2) | SBERT | SBERT + rerank |
|---|---|---|---|---|---|
| test_1255130000003 | The dress inspired millions of hashtags . | SUPPORTS | 2 | 1 | 1 |
| test_394050000004 | Lou Gehrig was a baseball player who defended the area near the second base . | SUPPORTS | 1 | 1 | 1 |
| test_194469 | Tilda Swinton is only Scottish . | REFUTES | 2 | 2 | 2 |
| test_1947750000004 | Larry the Cable Guy series finale aired on the month of August . | REFUTES | 4 | 5 | 2 |
| synthetic_nei_9 | Basildon is far away from England . | NOT ENOUGH INFO | no gold evidence | no gold evidence | no gold evidence |

## test_1255130000003 — gold label: SUPPORTS

> The dress inspired millions of hashtags .

Gold evidence: `ev_292`

**SBERT + FAISS** (score = cosine similarity)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 0.791 | ev_292 | ✓ | In the first week after the surfacing of the image alone, more than 10 million tweets mentioned the dress, using hashtags such as # thedress, # whitea |
| 2 | 0.719 | ev_558 |  | The dress was never popular on Twitter and could not inspire hashtags. |
| 3 | 0.356 | ev_55 |  | As of April 2017, she is one of the top 15 most followed celebrities on Instagram. |
| 4 | 0.314 | ev_456 |  | She has appeared on the cover of fashion magazines such as Vogue, Harper's Bazaar, Glamour, and Elle. |
| 5 | 0.300 | ev_573 |  | The genre was influential in Britain in shaping politics, national policies and popular perceptions in the years leading up to the First World War, an |

**SBERT + cross-encoder rerank** (score = rerank relevance; SBERT rank in brackets)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 0.997 (was 1) | ev_292 | ✓ | In the first week after the surfacing of the image alone, more than 10 million tweets mentioned the dress, using hashtags such as # thedress, # whitea |
| 2 | 0.996 (was 2) | ev_558 |  | The dress was never popular on Twitter and could not inspire hashtags. |
| 3 | 0.000 (was 6) | ev_293 |  | It achieved worldwide success by topping the charts in a variety of markets, ultimately selling 12 million copies worldwide, thus becoming one of the  |
| 4 | 0.000 (was 5) | ev_573 |  | The genre was influential in Britain in shaping politics, national policies and popular perceptions in the years leading up to the First World War, an |
| 5 | 0.000 (was 4) | ev_456 |  | She has appeared on the cover of fashion magazines such as Vogue, Harper's Bazaar, Glamour, and Elle. |

## test_394050000004 — gold label: SUPPORTS

> Lou Gehrig was a baseball player who defended the area near the second base .

Gold evidence: `ev_245`

**SBERT + FAISS** (score = cosine similarity)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 0.556 | ev_245 | ✓ | Henry Louis " Buster " Gehrig (born Heinrich Ludwig Gehrig; June 19, 1903June 2, 1941) was an American baseball second baseman who played 17 seasons i |
| 2 | 0.543 | ev_244 |  | Henry Louis " Buster " Gehrig (born Heinrich Ludwig Gehrig; June 19, 1903June 2, 1941) was an American baseball first baseman who played 17 seasons in |
| 3 | 0.378 | ev_229 |  | He played for the Giants and no other team. |
| 4 | 0.313 | ev_550 |  | The award was first given during the 1996 ceremony and since its conception, Omar Epps and Joe Morton hold the record for the most wins with three eac |
| 5 | 0.283 | ev_316 |  | It was founded as the Golden State Comic Book Convention in 1970 by a group of San Diegans that included Shel Dorf, Richard Alf, Ken Krueger and Mike  |

**SBERT + cross-encoder rerank** (score = rerank relevance; SBERT rank in brackets)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 0.976 (was 1) | ev_245 | ✓ | Henry Louis " Buster " Gehrig (born Heinrich Ludwig Gehrig; June 19, 1903June 2, 1941) was an American baseball second baseman who played 17 seasons i |
| 2 | 0.923 (was 2) | ev_244 |  | Henry Louis " Buster " Gehrig (born Heinrich Ludwig Gehrig; June 19, 1903June 2, 1941) was an American baseball first baseman who played 17 seasons in |
| 3 | 0.003 (was 3) | ev_229 |  | He played for the Giants and no other team. |
| 4 | 0.000 (was 13) | ev_235 |  | He was court martialed in 1971 for his role in the My Lai Massacre, but acquitted the same year. |
| 5 | 0.000 (was 16) | ev_230 |  | He played point guard for the Lakers for 13 seasons. |

## test_194469 — gold label: REFUTES

> Tilda Swinton is only Scottish .

Gold evidence: `ev_333`

**SBERT + FAISS** (score = cosine similarity)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 0.618 | ev_334 |  | Katherine Matilda " Tilda " Swinton (born 5 November 1960) is a Scottish actress, performance artist, model, and fashion muse, known for her roles in  |
| 2 | 0.585 | ev_333 | ✓ | Katherine Matilda " Tilda " Swinton (born 5 November 1960) is a British actress, performance artist, model, and fashion muse, known for her roles in i |
| 3 | 0.372 | ev_141 |  | Danielle Cormack (born 26 December 1970) is a stage and screen actress from New Zealand. |
| 4 | 0.372 | ev_142 |  | Danielle Cormack (born 26 May 1970) is a stage and screen actress from New Zealand. |
| 5 | 0.372 | ev_454 |  | She also avoided acting in the films The Cat in the Hat, Old Dogs, and Broken Bridges. |

**SBERT + cross-encoder rerank** (score = rerank relevance; SBERT rank in brackets)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 0.999 (was 1) | ev_334 |  | Katherine Matilda " Tilda " Swinton (born 5 November 1960) is a Scottish actress, performance artist, model, and fashion muse, known for her roles in  |
| 2 | 0.983 (was 2) | ev_333 | ✓ | Katherine Matilda " Tilda " Swinton (born 5 November 1960) is a British actress, performance artist, model, and fashion muse, known for her roles in i |
| 3 | 0.000 (was 18) | ev_218 |  | He always opposed a single parliament for England and Scotland. |
| 4 | 0.000 (was 10) | ev_475 |  | Sidse Babett Knudsen ([ˈsisə b̥ab̥ɛd̥ ˈkhnusn̩]; born 22 November 1968) is a Danish actress who works in theatre, television, and film. |
| 5 | 0.000 (was 9) | ev_474 |  | Sidse Babett Knudsen ([ˈsisə b̥ab̥ɛd̥ ˈkhnusn̩]; born 22 November 1948) is a Danish actress who works in theatre, television, and film after graduatin |

## test_1947750000004 — gold label: REFUTES

> Larry the Cable Guy series finale aired on the month of August .

Gold evidence: `ev_583`

**SBERT + FAISS** (score = cosine similarity)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 0.548 | ev_582 |  | The series finale aired August 28, 2013. |
| 2 | 0.537 | ev_400 |  | On Monday, July 11, 2011, the series began its eighth and final season, having finished its sixth season as cable's highest rated drama. |
| 3 | 0.512 | ev_584 |  | The series finale aired on May 14, 2012. |
| 4 | 0.509 | ev_585 |  | The series finale aired on May 14, 2013. |
| 5 | 0.504 | ev_583 | ✓ | The series finale aired September 28, 2013. |

**SBERT + cross-encoder rerank** (score = rerank relevance; SBERT rank in brackets)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 0.979 (was 1) | ev_582 |  | The series finale aired August 28, 2013. |
| 2 | 0.613 (was 5) | ev_583 | ✓ | The series finale aired September 28, 2013. |
| 3 | 0.528 (was 3) | ev_584 |  | The series finale aired on May 14, 2012. |
| 4 | 0.502 (was 4) | ev_585 |  | The series finale aired on May 14, 2013. |
| 5 | 0.141 (was 6) | ev_401 |  | On Monday, July 11, 2011, the series began its seventh and final season, having finished its sixth season as cable's highest rated drama. |

## synthetic_nei_9 — gold label: NOT ENOUGH INFO

> Basildon is far away from England .

Gold evidence: none (NOT ENOUGH INFO)

**SBERT + FAISS** (score = cosine similarity)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 0.523 | ev_71 |  | Basildon [ˈbæzɪldən] is the largest town in the borough of Basildon in the county of Essex, England. |
| 2 | 0.419 | ev_72 |  | Basildon [ˈbæzɪldən] is the largest town in the borough of New York in the United States of America. |
| 3 | 0.410 | ev_73 |  | Basildon [ˈbæzɪldən] is the latest hit of Lady Gaga. |
| 4 | 0.381 | ev_574 |  | The genre was never influential in Britain. |
| 5 | 0.330 | ev_385 |  | Middle-earth is a real place in the country of Germany. |

**SBERT + cross-encoder rerank** (score = rerank relevance; SBERT rank in brackets)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 0.995 (was 1) | ev_71 |  | Basildon [ˈbæzɪldən] is the largest town in the borough of Basildon in the county of Essex, England. |
| 2 | 0.801 (was 2) | ev_72 |  | Basildon [ˈbæzɪldən] is the largest town in the borough of New York in the United States of America. |
| 3 | 0.026 (was 3) | ev_73 |  | Basildon [ˈbæzɪldən] is the latest hit of Lady Gaga. |
| 4 | 0.000 (was 8) | ev_261 |  | Historically, the English population is descended from several peoples -- the earlier Britons (or Brythons) and the Germanic tribes that settled in Br |
| 5 | 0.000 (was 7) | ev_218 |  | He always opposed a single parliament for England and Scotland. |
