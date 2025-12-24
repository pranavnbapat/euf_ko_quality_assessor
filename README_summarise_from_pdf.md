## How to run:

```python
python summarise_from_pdf.py --input input/your_json.json 
--url-field "@id" --pages-per-chunk 3 --max-pages 200 --min-text-chars-for-no-image 100
```

## Output will be written to:
```
outputs/<input_stem>_visioned.json
```