# Lane 2 — sanity check on 5 sample claims (top-5)

## test_1255130000003 — gold label: SUPPORTS

> The dress inspired millions of hashtags .

Gold evidence: `ev_292`

**BM25** — gold in top-5: yes (rank 2)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 22.211 | ev_558 |  | The dress was never popular on Twitter and could not inspire hashtags. |
| 2 | 15.406 | ev_292 | ✓ | In the first week after the surfacing of the image alone, more than 10 million tweets mentioned the dress, using hashtags such as # thedress, # whitea |
| 3 | 7.097 | ev_407 |  | Pearl Jam is a Canadian type of dressing. |
| 4 | 4.999 | ev_151 |  | Despite of evolving for more than 50 million years, the horse has always maintained its size. |
| 5 | 4.671 | ev_415 |  | Poseidon grossed $ 181,674 at the worldwide box office on a budget of $ 160 million. |

**TFIDF** — gold in top-5: yes (rank 2)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 0.508 | ev_558 |  | The dress was never popular on Twitter and could not inspire hashtags. |
| 2 | 0.263 | ev_292 | ✓ | In the first week after the surfacing of the image alone, more than 10 million tweets mentioned the dress, using hashtags such as # thedress, # whitea |
| 3 | 0.163 | ev_407 |  | Pearl Jam is a Canadian type of dressing. |
| 4 | 0.091 | ev_151 |  | Despite of evolving for more than 50 million years, the horse has always maintained its size. |
| 5 | 0.079 | ev_415 |  | Poseidon grossed $ 181,674 at the worldwide box office on a budget of $ 160 million. |

## test_394050000004 — gold label: SUPPORTS

> Lou Gehrig was a baseball player who defended the area near the second base .

Gold evidence: `ev_245`

**BM25** — gold in top-5: yes (rank 1)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 14.280 | ev_245 | ✓ | Henry Louis " Buster " Gehrig (born Heinrich Ludwig Gehrig; June 19, 1903June 2, 1941) was an American baseball second baseman who played 17 seasons i |
| 2 | 11.992 | ev_244 |  | Henry Louis " Buster " Gehrig (born Heinrich Ludwig Gehrig; June 19, 1903June 2, 1941) was an American baseball first baseman who played 17 seasons in |
| 3 | 5.224 | ev_273 |  | In 1986, O'Neal married professional tennis player John McEnroe, with whom she had three children. |
| 4 | 5.224 | ev_272 |  | In 1986, O'Neal married professional tennis player John McEnroe, with whom she had no children. |
| 5 | 5.224 | ev_140 |  | Danielle Cormack (born 26 December 1970) is a golf player from New Zealand. |

**TFIDF** — gold in top-5: yes (rank 1)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 0.226 | ev_245 | ✓ | Henry Louis " Buster " Gehrig (born Heinrich Ludwig Gehrig; June 19, 1903June 2, 1941) was an American baseball second baseman who played 17 seasons i |
| 2 | 0.200 | ev_244 |  | Henry Louis " Buster " Gehrig (born Heinrich Ludwig Gehrig; June 19, 1903June 2, 1941) was an American baseball first baseman who played 17 seasons in |
| 3 | 0.080 | ev_140 |  | Danielle Cormack (born 26 December 1970) is a golf player from New Zealand. |
| 4 | 0.077 | ev_273 |  | In 1986, O'Neal married professional tennis player John McEnroe, with whom she had three children. |
| 5 | 0.077 | ev_272 |  | In 1986, O'Neal married professional tennis player John McEnroe, with whom she had no children. |

## test_194469 — gold label: REFUTES

> Tilda Swinton is only Scottish .

Gold evidence: `ev_333`

**BM25** — gold in top-5: yes (rank 2)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 14.710 | ev_334 |  | Katherine Matilda " Tilda " Swinton (born 5 November 1960) is a Scottish actress, performance artist, model, and fashion muse, known for her roles in  |
| 2 | 9.514 | ev_333 | ✓ | Katherine Matilda " Tilda " Swinton (born 5 November 1960) is a British actress, performance artist, model, and fashion muse, known for her roles in i |
| 3 | 0.000 | ev_632 |  | YouTube is an American video-sharing website headquartered in San Bruno, California. |
| 4 | 0.000 | ev_633 |  | " Hot Right Now " is a single by British drum and bass producer DJ Fresh, released as the second single from his second studio album, Kryptonite. |
| 5 | 0.000 | ev_634 |  | " Hot Right Now " is a single by British drum and bass producer DJ Fresh, released as the second single from his third studio album, Nextlevelism. |

**TFIDF** — gold in top-5: yes (rank 2)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 0.355 | ev_334 |  | Katherine Matilda " Tilda " Swinton (born 5 November 1960) is a Scottish actress, performance artist, model, and fashion muse, known for her roles in  |
| 2 | 0.261 | ev_333 | ✓ | Katherine Matilda " Tilda " Swinton (born 5 November 1960) is a British actress, performance artist, model, and fashion muse, known for her roles in i |
| 3 | 0.000 | ev_632 |  | YouTube is an American video-sharing website headquartered in San Bruno, California. |
| 4 | 0.000 | ev_633 |  | " Hot Right Now " is a single by British drum and bass producer DJ Fresh, released as the second single from his second studio album, Kryptonite. |
| 5 | 0.000 | ev_634 |  | " Hot Right Now " is a single by British drum and bass producer DJ Fresh, released as the second single from his third studio album, Nextlevelism. |

## test_1947750000004 — gold label: REFUTES

> Larry the Cable Guy series finale aired on the month of August .

Gold evidence: `ev_583`

**BM25** — gold in top-5: yes (rank 4)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 19.282 | ev_582 |  | The series finale aired August 28, 2013. |
| 2 | 14.692 | ev_584 |  | The series finale aired on May 14, 2012. |
| 3 | 14.692 | ev_585 |  | The series finale aired on May 14, 2013. |
| 4 | 14.137 | ev_583 | ✓ | The series finale aired September 28, 2013. |
| 5 | 10.167 | ev_401 |  | On Monday, July 11, 2011, the series began its seventh and final season, having finished its sixth season as cable 's highest rated drama. |

**TFIDF** — gold in top-5: yes (rank 4)

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 0.480 | ev_582 |  | The series finale aired August 28, 2013. |
| 2 | 0.458 | ev_585 |  | The series finale aired on May 14, 2013. |
| 3 | 0.450 | ev_584 |  | The series finale aired on May 14, 2012. |
| 4 | 0.409 | ev_583 | ✓ | The series finale aired September 28, 2013. |
| 5 | 0.113 | ev_400 |  | On Monday, July 11, 2011, the series began its eighth and final season, having finished its sixth season as cable 's highest rated drama. |

## synthetic_nei_9 — gold label: NOT ENOUGH INFO

> Basildon is far away from England .

Gold evidence: none (NOT ENOUGH INFO)

**BM25** — gold in top-5: no

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 12.581 | ev_71 |  | Basildon [ˈbæzɪldən] is the largest town in the borough of Basildon in the county of Essex, England. |
| 2 | 6.478 | ev_218 |  | He always opposed a single parliament for England and Scotland. |
| 3 | 6.234 | ev_234 |  | He was a major advocate of a single parliament for England and Scotland. |
| 4 | 6.134 | ev_73 |  | Basildon [ˈbæzɪldən] is the latest hit of Lady Gaga. |
| 5 | 5.933 | ev_545 |  | The Rock has played the title role in all of the Riddick-based films and video games so far. |

**TFIDF** — gold in top-5: no

| rank | score | evidence_id | gold? | text |
|---|---|---|---|---|
| 1 | 0.259 | ev_71 |  | Basildon [ˈbæzɪldən] is the largest town in the borough of Basildon in the county of Essex, England. |
| 2 | 0.143 | ev_218 |  | He always opposed a single parliament for England and Scotland. |
| 3 | 0.130 | ev_234 |  | He was a major advocate of a single parliament for England and Scotland. |
| 4 | 0.123 | ev_545 |  | The Rock has played the title role in all of the Riddick-based films and video games so far. |
| 5 | 0.122 | ev_73 |  | Basildon [ˈbæzɪldən] is the latest hit of Lady Gaga. |
