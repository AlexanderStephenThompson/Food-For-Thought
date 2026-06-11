# Merge Decision Spot-Check Sample

All 94 borderline cases resolved: 47 merge / 41 preserve / 6 merge_into.
38 contested -> skeptic review; 15 -> tiebreaker. 27 finals differ from the pure-statistics suggestion.

This sample = all 15 tiebreaker cases (where judge and skeptic disagreed) plus 5 skeptic-upheld
decisions that went against the statistical suggestion — the highest-risk subset.
Each is one line in lexicons/merge_decisions.jsonl and individually reversible.

## 1. black lentil  vs  lentil
**Final: preserve** (suggested: merge; route: tiebreaker)

judge: merge | skeptic: preserve | final: preserve — Verified against raw/kaggle/train.json: the skeptic's facts hold. 5 of 6 'black lentil' recipes are Indian whole-urad (Vigna mungo) uses — three dal makhani (ids 10274, 25236, 10784, all pairing it with kidney beans + cream/butter) where black lentil is the dish-defining ingredient, one tadka tempering (35150), one urad preparation (43680); only the Greek salad (11286) is a true Lens culinaris beluga varietal. The judge's 'same culinary role varietal' premise is wrong for the majority of occurrences, so the identity test fails decisively: swapping generic lentils in changes the dish. Stats (ratio 0.876 at n=6, null95 0.363 bits) are underpowered, but the shift is coherent (83% indian, lift 11.0 vs base 6.2, zero moroccan/italian/french mass), not a source artifact. Cost asymmetry favors preserve: a wrong merge irreversibly dilutes an 11x indian-lift ingredient into a base with ~33% moroccan/italian/french mass, while preserve retains back-off via the parent link. merge_into 'split black lentils' correctly rejected — that key exists (14 recipes) but covers the split tempering form, distinct from whole urad, and the Greek recipe is not urad.

## 2. black rice vinegar  vs  rice vinegar
**Final: merge_into:black vinegar** (suggested: merge; route: tiebreaker)

judge: preserve | skeptic: merge_into:black vinegar | final: merge_into:black vinegar — 'Black rice vinegar' is Chinkiang black vinegar, the same ingredient as the existing vocabulary key 'black vinegar' (verified as base of chinese_black_vinegar__vs__black_vinegar, n=40, 90% Chinese, lift 13.4). The judge correctly rejected merging into pan-Asian 'rice vinegar' (9/9 Chinese, p~1.7e-4 under the base) but chose the wrong remedy: preserve would strand an n=9 node with a misleading pan-Asian backoff parent, while aliasing into 'black vinegar' consolidates ~92 occurrences of the identical ingredient with the ~90-100% Chinese signal fully intact. Distributions of variant and target are statistically indistinguishable, so the merge destroys no signal.

## 3. chinese rice vinegar  vs  rice vinegar
**Final: preserve** (suggested: merge; route: tiebreaker)

judge: merge | skeptic: preserve | final: preserve — Judge's premise is factually wrong: ratio 1.35 means jsd (0.4121) EXCEEDS null95 (0.3059) at n=8 — the variant clears the null, just not the 2x bar. Verified in raw train.json: 8/8 recipes are chinese (p ~ 4.6e-4 under the base's 38.23% share), a perfectly coherent concentration in exactly the cuisine the string names, vs a diffuse pan-Asian base — real cuisine signal, which alone satisfies the preserve condition. Project precedent in lexicons/named_varieties.json already preserves 'white rice vinegar' (n=9, 78% chinese) and 'chinese red vinegar', explicitly warning that default-merge silently deletes near-perfect cuisine markers; 100% at n=8 is strictly stronger. Cost asymmetry seals it: merge irreversibly destroys a near-perfect Chinese marker; preserve costs only trivial fragmentation recoverable via the parent link to 'rice vinegar'.

## 4. condensed tomato soup  vs  tomato soup
**Final: merge** (suggested: preserve; route: tiebreaker)

judge: preserve | skeptic: merge | final: merge — Stats are sub-threshold (1.62x null95) with concordant Mexican-dominant profiles on both sides (shares 63.6% vs 53.8%, lifts 3.9 vs 3.3, 4/5 top cuisines overlapping); the lone Italian shift is 1-vs-6 recipes at n=11/n=26 — small-sample noise with no culinary story. Identity test is decisive for merge: as an ingredient line in US recipes, 'tomato soup' already denotes the condensed canned product (the base's Mexican/southern casserole-base profile confirms the identical role), so 'condensed' is a redundant descriptor, not a distinct ingredient. Rule 3's preserve tiebreaker never activates because rule 2 resolves it; merging yields a stronger n=37 Mexican-leaning signal.

## 5. dark rum  vs  rum
**Final: preserve** (suggested: merge; route: tiebreaker)

judge: merge | skeptic: preserve | final: preserve — Verified all skeptic claims against the repo. (1) forced_merge_overrides.json explicitly documents "light/dark rum must PRESERVE" and always_merge_patterns.json records light-vs-dark rum JSD 0.339 bits as an ingredient identity — project policy already treats dark rum as identity-bearing, and merging it while light rum stays separate would corrupt the 'rum' base asymmetrically. (2) Raw train.json confirms the French/British tilt is culinarily coherent: British dark-rum recipes are Christmas pudding/fruitcake/hard sauce (candied peel, currants, demerara; id 14944 is hard sauce), French ones are baba au rhum/savarin desserts (id 24142), Mexican share halves (10.4%->5.2%) as light-rum contexts drop out — a directional story across diverse recipes, not a source artifact. (3) JSD 0.0588 > null95 0.0383 at ratio 1.53 is borderline, and the framework says borderline stats defer to identity and cost asymmetry: variant n=116 > base n=77, so a wrong merge irreversibly dilutes the larger sample, while preserve retains parent back-off. Recipe 45501 lists 'rum' and 'dark rum' as separate ingredients. Side with the skeptic: preserve.

## 6. dried fig  vs  fig
**Final: merge** (suggested: preserve; route: tiebreaker)

judge: preserve | skeptic: merge | final: merge — Verified against raw train.json: no 'fresh figs' string exists anywhere in the corpus, and the base 'figs' (n=33: italian 10, british 5, moroccan 5, french 5, indian 3, spanish 2) is itself a dried-fig profile (figgy pudding, tagines, anjeer), so the judge's fresh/dried role distinction was never encoded. The claimed Italian/Spanish/Greek shift is count noise at n=26 vs 33: Spanish is exactly 2 recipes on both sides (hidden by the base's top-5 cutoff), Greek 2 vs 0, Italian an ~2-recipe swing; the largest shift (British 5 vs 1) is itself dried-fig usage. jsd_to_null_ratio 1.51 is below the 2x bar, and swapping 'figs' into the example recipes (Spanish almond-fig dish 3159, British fig tart 4054, Italian fig fruitcake 6434) yields the same dish and cuisine read. Stats and identity both point merge, so the rule-3 preserve tiebreak does not trigger; merging consolidates a sparse 59-recipe family with matching distributions.

## 7. dried wood ear mushroom  vs  wood ear mushroom
**Final: merge** (suggested: preserve; route: tiebreaker)

judge: preserve | skeptic: merge | final: merge — Verified in raw train.json: the variant's 6 occurrences are only 4 distinct recipes (45993/4127 and 2461/5567 are exact duplicates, identical ingredient lists and cuisine), so the 4.2x jsd_to_null_ratio is inflated by a null computed at n=6. Deduped, 4/4 Vietnamese is still unlikely by chance (p~0.0009) but borderline given one identical house phrasing across all occurrences and evident source correlation. Identity is decisive: wood ear is sold dried and rehydrated in Chinese, Vietnamese, and Japanese cooking alike, so 'dried' carries no differential cuisine story — swapping the base in changes nothing about the dish. The base already lifts Vietnamese 8.3x (5/29 recipes), so merging consolidates rather than destroys the Vietnamese signal (post-merge ~31% Vietnamese share). Siding with the skeptic.

## 8. hot red pepper flake  vs  red pepper flake
**Final: preserve** (suggested: preserve; route: tiebreaker)

judge: merge | skeptic: preserve | final: preserve — Ratio 2.44x null95 at n=171 with a coherent Mediterranean-vs-East-Asian shift: italian 30.8%->44.4% (+3.9 SE), spanish enters at 3.06x lift, while chinese drops 14.7%->9.9% and korean (3.80x lift) + thai (2.01x lift) — 15.7% of the base — fall out of the variant's top-5. The judge's "only shift is italian deepening" misreads the table. The phrasing has a culinary story ("hot red pepper flakes" is Italian-American peperoncino phrasing; East-Asian recipes use gochugaru or plain "red pepper flakes"), so the rule-4 artifact exemption fails. Same-pantry-product identity does not override above-threshold cuisine signal (merge UNLESS signal OR identity), and cost asymmetry breaks any residual tie toward preserve, with the parent link to red pepper flakes recovering pooled counts.

## 9. hungarian sweet paprika  vs  sweet paprika
**Final: merge** (suggested: preserve; route: tiebreaker)

judge: preserve | skeptic: merge | final: merge — Skeptic's factual correction verified against train.json: russian is present in the base (3/134, share 2.2%, lift 1.82), so the judge's 'absent from base' premise is false. The russian bump (4/37, Fisher p=0.040 raw) dies under 20-cuisine multiplicity (p=0.80), consistent with the sub-threshold 1.52x jsd_to_null_ratio. The true JSD drivers are artifact-shaped: the spanish near-doubling comes from saffron paellas/shrimp dishes where 'Hungarian sweet paprika' is authorial style, not cuisine signal (Spanish cooking uses pimenton), and the cajun/southern_us collapse has no culinary story. Identity test is decisive: Hungarian sweet paprika (edesnemes) is the canonical sweet paprika — a provenance label, not a process/heat variety like smoked or hot paprika, so the paprika-variety preserve precedent does not apply. The variant's 4 russian recipes are genuine goulash/stroganoff dishes, but merging reinforces the base's existing russian association rather than destroying signal. Steps 1 and 2 both say merge, so the step-3 torn-tiebreak never activates. Merge 'hungarian sweet paprika' into 'sweet paprika'.

## 10. italian herb  vs  herb
**Final: merge_into:italian seasoning** (suggested: merge; route: tiebreaker)

judge: preserve | skeptic: merge_into:italian seasoning | final: merge_into:italian seasoning — Both reviewers correctly reject merging into 'herbs' (7/8 Italian vs base 26.9%, binomial p~6e-4; ratio 1.11 is purely an n=8 null artifact). But the skeptic's sibling claim verifies fully against raw train.json: 'italian seasoning' exact n=395 at 87.8% Italian, statistically indistinguishable from the variant's 87.5%; base 'seasoning' n=137 at 8.76% Italian makes that pair a clean L2 auto-preserve (absent from the 94-entry queue), 'italian' is a variant modifier token, and no lexicon aliases 'italian seasoning' away — so the key exists. Example recipes (4057, 6870, 8373) use 'Italian herbs' as a dried pantry blend in pasta/sausage dishes, i.e., the same commercial product as Italian seasoning. The judge's preserve would wire an n=8 key to the WRONG parent ('herbs', fresh-herb Vietnamese/French profile), making preserve the riskier option here. Aliasing into 'italian seasoning' retains the Italian signal at zero cost. Minor correction to the skeptic: southern_us (n=16), not cajun_creole (n=7), is second for 'italian seasoning' — immaterial.

## 11. italian parsley leaf  vs  parsley leaf
**Final: preserve** (suggested: preserve; route: tiebreaker)

judge: merge | skeptic: preserve | final: preserve — Statistics show a real, coherent shift, not noise: Italian share 41.2% -> 59.2% (lift 2.09 -> 3.00) at n=125 is ~4.1 SEs above the base rate, and JSD is 1.8x the null95 (merge-lean requires <1). The divergence concentrates in exactly the cuisine named, across ~74 recipes, ruling out a one-source artifact. The identity test is split (same dish in the kitchen, but the variant reads more Italian per the protocol's own clause), so cost asymmetry decides: a wrong merge irreversibly dilutes the variant into a ~48% pooled blend, while preserve with the parent link to 'fresh parsley leaves' (n=192 remains healthy) loses almost nothing. No sibling 'flat leaf parsley' key exists in the queue, so merge_into is inapplicable. Pipeline suggestion was also preserve.

## 12. italian plum tomato  vs  plum tomato
**Final: preserve** (suggested: preserve; route: tiebreaker)

judge: merge | skeptic: preserve | final: preserve — Verified in train.json: 26/34 variant recipes are Italian vs 39.6% base rate (binomial p=1.4e-05), with significant Mexican depletion (2/34 vs ~7.3 expected, p=0.014) and elevated Greek — a Mediterranean-coherent shift the omnibus JSD ratio (1.50) understates. Artifact check passes: 25/26 unique ingredient sets, 1 near-duplicate pair, diverse classic-Italian co-ingredients, so not a one-source quirk. The judge correctly refutes the identity prong (same physical tomato) but the rule is disjunctive; 'Italian plum tomatoes' as the standard term for canned San Marzano-style tomatoes carries real, verified cuisine signal. Cost asymmetry and the pipeline's own statistical suggestion also favor preserve; the parent link to 'plum tomatoes' recovers any count fragmentation.

## 13. italian seasoned dry bread crumb  vs  dry bread crumb
**Final: merge_into:seasoned bread crumbs** (suggested: merge; route: tiebreaker)

judge: preserve | skeptic: merge_into:seasoned bread crumbs | final: merge_into:seasoned bread crumbs — All skeptic claims independently verified against train.json and the pipeline's own divergence harness. Both reviewers correctly reject plain merge: 10/11 Italian vs a 41.6%-Italian base is binomial p~0.001 real signal that the n=11 JSD null (0.315 bits) cannot see. But the variant is not a distinct ingredient: vs sibling 'seasoned bread crumbs' (n=98, 72.4% Italian) it scores jsd=0.0994 @ 0.51x null95 — a random sample of the sibling — and 'Italian seasoned breadcrumbs' (n=30, 76.7% Italian) sits at 0.98x null against the same sibling, confirming one product family (US seasoned crumbs are dry and Italian-seasoned by definition). Sibling key existence verified: n=98, jsd=0.1640 @ 2.78x null95 vs 'bread crumbs' (clean L2 preserve, outside both borderline bands), no crumb entries in any merge lexicon, absent from the 94-entry review queue. The judge's preserve would create an n=11 fragment whose parent backoff points at the diluted 'dry bread crumbs' distribution; aliasing into 'seasoned bread crumbs' keeps the full 72%-Italian signal with no fragmentation.

## 14. jamaican rum  vs  rum
**Final: preserve** (suggested: merge; route: tiebreaker)

judge: merge | skeptic: preserve | final: preserve — Verified in train.json: all 5 'jamaican rum' recipes are jamaican; base 'rum' is 19/77 jamaican (24.7%). Exact test P(5/5 | 0.247) = 0.0009 (0.018 Bonferroni x20) rejects "random sample of base." The judge misread ratio 1.055 — it means JSD exceeds the 95th-percentile null, not that the variant looks random; the generic gate is just underpowered at n=5. Merging collapses P(jamaican|ingredient) from 1.0 to 0.293 (lift 76x -> ~22x) on a rare (~1.3% prior) cuisine — real signal loss in a blend classifier. Identity is shared (same dish role), but the rule is disjunctive (signal OR identity) and the demonym is an empirically attached authenticity marker with a coherent culinary story, not a source artifact. Cost asymmetry settles residual doubt: parent link to 'rum' retains backoff at near-zero cost.

## 15. split black lentil  vs  black lentil
**Final: merge_into:urad dal** (suggested: merge; route: tiebreaker)

judge: merge | skeptic: merge_into:urad dal | final: merge_into:urad dal — All skeptic claims verified against the queue, lexicons, and train.json. (1) The base 'black lentil' (n=6, beluga-contaminated via 1 Greek recipe) is itself queued at line 7 as black_lentil__vs__lentil with ratio 0.8756 and suggested merge into generic 'lentils' (47% Indian) — upholding the judge would chain 14 recipes at 100% Indian / lift 13.2 into a multi-cuisine pool, irreversibly destroying signal. (2) The judge misread ratio 1.0: it is the null95 boundary (borderline), not the <1 merge zone. (3) The pairing is a modifier_strip.json artifact ('split' at line 47), not a real culinary parent. (4) 'urad dal' exists in the corpus at n=68, 97.1% Indian (66/68), lowercase, untouched by lexicon rules — a certain vocabulary key. 'Split black lentils' is unambiguously split urad dal; aliasing in loses zero cuisine signal (100% Indian into 97% Indian), consolidates n=82 on the culinarily correct ingredient, and beats preserve, whose parent backoff would target a dissolving base.

## 16. baby bok choy  vs  bok choy
**Final: merge** (suggested: preserve; route: judge+skeptic-upheld)

Despite a 2.22x ratio, the Thai-up/Filipino-down shift has no culinary story and baby bok choy makes the identical dish as bok choy — the rubric's own canonical merge example.

## 17. black cardamom pod  vs  cardamom pod
**Final: preserve** (suggested: merge; route: judge+skeptic-upheld)

Statistics look null (0.85x) but black cardamom is a different species with a smoky flavor not interchangeable with green cardamom, and a wrong preserve at n=10 costs almost nothing.

## 18. black cumin seed  vs  cumin seed
**Final: preserve** (suggested: merge; route: judge+skeptic-upheld)

Black cumin (kala jeera) is a distinct spice from regular cumin with 100% Indian usage; identity is decisive and fragmenting n=5 is cheap versus an irreversible merge.

## 19. candied orange peel  vs  orange peel
**Final: preserve** (suggested: merge; route: judge+skeptic-upheld)

Candied orange peel is a confection with a distinct baking role (panettone, fruitcake) and an Italian-shifted distribution, so swapping in fresh grated peel would change the dish.

## 20. dried scallop  vs  scallop
**Final: preserve** (suggested: merge; route: judge+skeptic-upheld)

Identity is decisive despite borderline stats: dried scallops (conpoy) are a distinct Chinese umami seasoning that cannot be swapped for fresh scallops, and the 100% Chinese share at 14.9x lift backs this up.

