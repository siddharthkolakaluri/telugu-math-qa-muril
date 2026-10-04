# Test prompts — Telugu math QA

Manual test set for the fine-tuned MuRIL demo. Each entry gives the passage,
the question, the span you should expect, and what the case is probing.

Remember the constraint throughout: the model selects a span of the passage.
It can only return text that is already written there.

---

## A. Template cases — answer stated at the end

These match the MAWPS training format. Expect confident, correct spans. They
prove very little, which is the point: a rule returning the last number also
gets them right.

**A1**

> రాజు దగ్గర 15 పెన్సిళ్లు ఉన్నాయి. అతను 6 పెన్సిళ్లు పోగొట్టుకున్నాడు. సమాధానం 9.

Question: `సమాధానం ఏమిటి?`
Expect: **9**
*(Raju had 15 pencils, lost 6. The answer is 9.)*

**A2**

> ఒక తరగతిలో 24 మంది విద్యార్థులు ఉన్నారు. వారిలో 10 మంది అమ్మాయిలు. సమాధానం 14.

Question: `సమాధానం ఏమిటి?`
Expect: **14**
*(A class has 24 students, 10 are girls. The answer is 14.)*

---

## B. Comprehension cases — answer sits inside the passage

The real test. The answer is present but not at a fixed position, so the model
has to work out which span the question is asking about.

**B1**

> లత దగ్గర ముప్పై రూపాయలు ఉన్నాయి. ఆమె ఇరవై రూపాయలు ఖర్చు చేసింది. ఆమె దగ్గర ఇప్పుడు పది రూపాయలు మిగిలాయి.

Question: `లత దగ్గర ఇప్పుడు ఎంత డబ్బు మిగిలింది?`
Expect: **పది రూపాయలు**
*(Latha had thirty rupees, spent twenty, has ten left.)*

**B2**

> కిరణ్ ఉదయం ఏడు గంటలకు పాఠశాలకు వెళ్తాడు. అతను సాయంత్రం నాలుగు గంటలకు ఇంటికి తిరిగి వస్తాడు.

Question: `కిరణ్ ఎన్ని గంటలకు ఇంటికి వస్తాడు?`
Expect: **సాయంత్రం నాలుగు గంటలకు**
*(Kiran leaves for school at 7am, returns home at 4pm.)*

Note the trap: there are two times in the passage. Picking the *first* one is a
plausible failure and worth watching for.

---

## C. Distractor numbers — several candidates, one correct

**C1**

> ఒక దుకాణంలో 12 ఆపిల్లు, 8 అరటిపండ్లు, 5 మామిడిపండ్లు ఉన్నాయి.

Question: `అరటిపండ్లు ఎన్ని ఉన్నాయి?`
Expect: **8**
*(A shop has 12 apples, 8 bananas, 5 mangoes.)*

Run this one three times with the question changed to ask about apples, then
mangoes. If the span moves correctly each time, the model is reading the
question rather than defaulting to a position.

**C2**

> రవి 3 పుస్తకాలు, 7 నోట్‌బుక్‌లు, 2 పెన్నులు కొన్నాడు.

Question: `రవి ఎన్ని నోట్‌బుక్‌లు కొన్నాడు?`
Expect: **7**
*(Ravi bought 3 books, 7 notebooks, 2 pens.)*

---

## D. Non-numeric answers — Wikipedia-style, like IndicQA

This is closest to the data behind the 57.7 F1, and where multi-word spans make
boundary errors visible.

**D1**

> భారతదేశ రాజధాని న్యూఢిల్లీ. ఇది ఉత్తర భారతదేశంలో ఉంది.

Question: `భారతదేశ రాజధాని ఏది?`
Expect: **న్యూఢిల్లీ**
*(The capital of India is New Delhi. It is in north India.)*

**D2**

> తెలుగు భాష ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో మాట్లాడతారు.

Question: `తెలుగు ఏ రాష్ట్రాలలో మాట్లాడతారు?`
Expect: **ఆంధ్రప్రదేశ్ మరియు తెలంగాణ**
*(Telugu is spoken in Andhra Pradesh and Telangana.)*

A three-word answer. Watch whether the span stops cleanly or swallows
రాష్ట్రాలలో on the end — that overshoot is exactly the boundary error your
Qualitative Analysis describes, and it's what drives F1 above Exact Match.

---

## E. The limitation — answer is not in the passage

These should fail, and the failure is the useful result. Run at least one of
them before you demo the app to anyone, so the behaviour isn't a surprise.

**E1**

> రాము దగ్గర 10 ఆపిల్లు ఉన్నాయి. అతను 3 ఆపిల్లు ఇచ్చాడు.

Question: `ఇప్పుడు రాము దగ్గర ఎన్ని ఆపిల్లు మిగిలాయి?`
Expect: **no correct answer is possible.** The answer is 7, and 7 appears
nowhere in the passage, so the model cannot produce it. It will return some
other span — probably 10 or 3.
*(Ramu has 10 apples, gave away 3. How many remain?)*

**E2**

> ఒక పెట్టెలో 6 వరుసలు ఉన్నాయి. ప్రతి వరుసలో 4 వస్తువులు ఉన్నాయి.

Question: `మొత్తం ఎన్ని వస్తువులు ఉన్నాయి?`
Expect: **nothing correct.** 24 is not written in the passage. Multiplication
is not something an extractive model can do.
*(A box has 6 rows, each row has 4 items. How many items in total?)*

---

## F. Edge cases

**F1 — answer is the first word**

> న్యూఢిల్లీ భారతదేశ రాజధాని నగరం.

Question: `భారతదేశ రాజధాని ఏది?`
Expect: **న్యూఢిల్లీ** — tests a span starting at character 0.

**F2 — the same number twice**

> సీత 5 పెన్నులు కొన్నది. రాజు కూడా 5 పెన్నులు కొన్నాడు.

Question: `రాజు ఎన్ని పెన్నులు కొన్నాడు?`
Expect: **5**, from the *second* sentence. Open Model internals and check the
character offsets — if they point at the first 5, the model found the right
text for the wrong reason, which an F1 score alone would never reveal.

---

## What to record

For each case worth writing up, Model internals gives you:

- **Span, decoded** vs **Span, from passage** — if these differ, subword `##`
  artifacts are in play
- **Produced by** — `span` means the model, `pattern` means the fallback
  heuristic substituted a number and the answer is not the model's
- **Character offsets** — the only way to tell F2 apart
- **Span score** — useful for comparing cases, not as a probability

Group E is the set most worth a screenshot. It demonstrates the architectural
limit your paper's Limitations section describes, and it's more convincing
shown than asserted.
