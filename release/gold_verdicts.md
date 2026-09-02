# Hand audit of 283 machine-written labels

Method: 300 labeled reviews sampled with stratification (60 uninformative, 80 with
modifications, 80 with taste/texture flags, 80 plain informative), then checked by hand
against the original review text, field by field. 283 of 300 were fully judged (17 lost
to output truncation during the audit). Precision = of the labels the extractor
*asserted*, how many the review actually supports. Recall was not formally scored;
notable misses are listed at the end.

## Per-field precision

| field | asserted | precision | notes |
|---|---|---|---|
| informative | 283 | ~97% | every error was "should be true, labeled false" — conservative direction, harmless |
| made_as_written | ~14 | ~93% | reliable |
| difficulty | ~35 | ~94% | reliable |
| modifications | ~120 | ~90% | a few technique changes recorded as ingredients; rare empty-item artifacts |
| taste: bland | ~13 | ~92% | the most valuable negative signal — trustworthy |
| taste: salt / sweet / sour / spicy | ~18 | ~78% | moderately reliable |
| would_make_again | ~85 | ~70% | systematic bias: gushing five-star prose gets inferred as "will make again" without an explicit statement |
| audience | ~55 | ~73% | over-generalized: "my family" ≠ kids; "weekend project" ≠ weeknight |
| timing | 5 | ~40% | tiny sample and the *direction is often flipped* ("baked too long" labeled too_short) |
| taste: rich | ~45 | ~50% | **unreliable** — generic praise (awesome / delicious) becomes "rich"; it is a praise-intensity proxy |
| texture_issues | ~14 | ~36% | **worst field** — praise gets inverted ("so moist" → dry), categories blur (runny → soggy/greasy), and complaints about *other* recipes get attached to this one |

## Guidance for downstream use

- **Trust directly:** the informative filter, bland, modifications, difficulty,
  made_as_written, salt/sweet.
- **Discount:** would_make_again (over-asserted), audience.
- **Aggregate-only:** rich (praise proxy) and texture_issues — their recipe-level rates
  still correlate with ratings in the right direction, but no single label should be
  shown to a user as an explanation.
- timing needs directional few-shot examples in the next prompt revision.

## Notable recall misses (not scored)

Explicit "won't make this again" statements were sometimes left null; accidental
omissions ("forgot the sugar — still great") were missed as modifications; a few
meal-prep and picky-eater signals went unlabeled.
