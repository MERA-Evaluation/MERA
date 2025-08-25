### Автособираемая карточка датасета

Для автосборки карточки датасета, он должен быть оформлен в описанном в [инструкции](../docs/dataset_formatting.md) формате.

Запускать скрипт из корня, подставить название папки с датасетом вместо YOUR_DATASET:

```
python scripts/autocollect_docs.py "YOUR_DATASET"
```
В результате будут созданы следующие файлы:
- `YOUR_DATASET/dataset_meta.json`
- `YOUR_DATASET/README.md`
- `YOUR_DATASET/README_ru.md`
