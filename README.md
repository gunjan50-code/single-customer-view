# Single Customer View

Banks store the same customer over and over without realising it. This project
finds those duplicates and merges them into one clean record.

Python, pandas, scikit-learn, rapidfuzz, jellyfish, networkx, SQLite, Streamlit

## The problem

A woman opens a savings account at a branch in 2019 and a clerk types her name by
hand. In 2021 she applies for a credit card online and types it herself, slightly
differently. In 2023 she calls support and the agent writes down a nickname.

Here are five rows from this project's generated data, printed by
`python -m src.inspect_sample`. They are all the same woman:

| source | first | last | dob | email | phone | address | city | pincode |
|---|---|---|---|---|---|---|---|---|
| web_signup | Pillai | Lakshmi | 05/06/1997 | laks@yahoo.co.in | 8741347350 | 302 Sector 14 | Jaipur | 640535 |
| crm | Lakshmi | Pillai | 05 Jun 1997 | lakshmi.p@yahoo.co.in | 8741348350 | 302 Sector 14 | Jaipur | 640535 |
| billing | Lakshmi | Pkllai | 05 Jun 1997 | lakshmi.p@yahoo.co.in | 8741347350 | 302 Sector 14 | Jaipur | 640535 |
| branch | L. | Pillai | 05 Jun 1997 | lakshmi.p@yahoo.co.in | 087413-47350 | 62 Hospital Road | Visakhapatnam | 562456 |
| support | LAKSHMI | PILLAI | not provided | not provided | 87413 47350 | 302 SECTOR 14 | JAIPUR | 640535 |

The bank thinks it has five customers. It has one.

Every kind of damage this project has to survive is visible in those five rows.
The first record has the name the wrong way round. The third has a typo in the
surname. The fourth is initials only, and she had moved house by the time it was
created, so the address and city agree with nothing. The second has a mistyped
phone digit. The last is all caps with the date of birth and email missing
entirely. No two rows are identical, and no single field is reliable on its own.

That is not a cosmetic problem. Split across five records, her balance looks small
enough to stay under money laundering reporting thresholds. A credit check sees
one of her loans instead of all five. She gets five copies of every mailer. And
every customer count the bank reports is wrong.

Banks pay consultancies a lot of money to fix this. The industry calls it Master
Data Management, or a single customer view.

## The two things that make it hard

**They never look identical.** "Rajesh Kumar Sharma" and "R. K. Sharma" share
almost no characters. Comparing with `==` is useless, so I score how similar two
records are across 20 different measures and let a model decide what "similar
enough" means.

**There are far too many pairs.** 31,000 records give 482 million possible pairs.
At 10 million records it is 50 trillion, and the whole approach dies. So before
comparing anything carefully, a cheap filter throws away pairs that obviously are
not worth checking. That step, called blocking, removed 99.96 percent of the work
while keeping 96 percent of the real duplicates.

## What I built

Seven stages, each one a small script:

1. **Generate the data.** I invent 20,000 real people, then scatter damaged copies
   of them across six fake source systems. Because I control the damage, I know
   the right answer for every record, which gives me training data and an honest
   test set without paying anyone to label anything.

2. **Standardize.** Fix casing, punctuation, phone formats, six different date
   formats, and city spellings. Blore, Bangalore and BLR all become bengaluru.
   This alone collapsed 323 spellings down to 21 real cities.

3. **Blocking.** Five cheap keys, unioned. One of them ignores the name entirely,
   because a nickname like Rajesh becoming Raju destroys every name-based key.

4. **Features.** Turn each surviving pair into 20 numbers describing how similar
   they are. Missing data is handled explicitly: a missing phone is no evidence,
   which is different from a phone that disagrees.

5. **Train.** Logistic regression, picked over anything fancier because I can read
   one coefficient per feature and explain any decision it makes.

6. **Decide.** Not one threshold but two. Confident matches merge automatically,
   confident non-matches are rejected automatically, and everything in between
   goes to a human. This is the Fellegi-Sunter design from 1969, still what
   commercial tools use.

7. **Cluster and merge.** Pairs become a graph, connected groups become people,
   and survivorship rules pick which name and address survive. Every surviving
   field records which system it came from, because "the system merged it" is not
   an answer an auditor accepts.

There is also a Streamlit screen for working through the pairs the model refused
to decide, and an optional stage that sends only those ambiguous pairs to an LLM.

## Results

Measured on held-out people. The split is by person, not by pair, so nobody
appears on both sides of it.

| | |
|---|---|
| Possible pairs, reduced to | 482,000,000 to 172,329 |
| Real duplicates kept after blocking | 96.4 percent |
| Classifier F1 | 0.983 |
| Same, using a sensible hand-written rule instead | 0.419 |
| Decisions made without a human | 96.0 percent |
| Wrong merges | 3 |
| People reconstructed perfectly | 92.9 percent |
| 31,040 records collapsed into | 21,244 real people |

Runs end to end in about two minutes.

## Things I got wrong on the way

**My first blocking key was wasteful.** It generated 597,000 candidate pairs, 76
percent of all the work in the pipeline, and found only 107 duplicates that no
other key found. I changed it to require two name parts to agree instead of one.
That cut the work by 71 percent and cost 0.21 percent of recall.

**The phone number was cheating.** In my first data generator every person had a
unique phone that survived corruption perfectly, so the model just matched on
phone and scored 0.99. That is not a matcher, it is a lookup. I added mistyped
digits, and families who share a number, which broke the shortcut and forced the
model to actually use the other 19 features.

**Calibration did not work.** I assumed adding probability calibration would make
the scores trustworthy. I measured it, and it barely moved: all methods stay about
11 percent off in the middle of the range. There is simply not enough data there
to learn from. The honest conclusion is that the model is least reliable exactly
where it is least certain, which is the argument for sending those pairs to a
person instead of guessing.

## Limitations

The data is synthetic, so these numbers are optimistic. I designed the features
knowing exactly what damage I had applied. Real entity resolution lands nearer
0.85 to 0.95 F1.

The LLM stage has never actually run. The code and prompts are finished and the
cost estimates come from real measured prompt lengths, but I have not reported any
accuracy for it because I have not measured any.

Pairwise recall of 0.885 is the real weakness. 3.6 percent is lost in blocking and
cannot be recovered later. The rest sits in the deliberately cautious merge
threshold, on the grounds that wrongly merging two people's bank accounts is far
worse than missing a duplicate.

The runaway cluster guards never triggered on this data. They are there as a
safety net, not because they saved anything.

## Running it

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Then have a look at the problem itself, and the review queue:

```bash
python -m src.inspect_sample
streamlit run app.py
```

Generated data is gitignored, but the random seed is fixed, so a fresh clone
reproduces every number above.

