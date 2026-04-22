## POLLUX Instructions Example

This directory contains a minimal open-generation example built from the
`ai-forever/POLLUX-instructions` dataset.

Source prompt:
- dataset: `ai-forever/POLLUX-instructions`
- split: `train`
- `prompt_id`: `0`
- instruction:
  `Составь мне план научного доклада об измерении содержания метана в испарениях над морем Лаптевых.`

Source criterion:
- dataset: `ai-forever/POLLUX-criteria`
- task subtype: `Составить план текста`
- domain: `Научный`
- criterion name: `Глубина проработки ответа`

The example keeps only one criterion on purpose so that the task stays short and
readable in the repository. Real tasks should usually copy the full criteria set
from `POLLUX-instructions`.
