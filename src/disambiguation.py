import os
import json
import csv
import requests
from evaluate import evaluate
from operator import itemgetter

class Disambiguator:
    def __init__(self, 
                 cache_path="predictions/cache/wikidata_cache.json",
                 max_cache_size=10000):
        self.cache_path = cache_path
        self.max_cache_size = max_cache_size
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self.cache = self._load_cache()
        # Add counters for tracking disambiguation operations
        self.total_entities = 0
        self.cache_hits = 0
        self.api_calls = 0
        self.successful_disambs = 0

    def _load_cache(self):
        if os.path.isfile(self.cache_path):
            with open(self.cache_path, "r") as f:
                cache_data = json.load(f)
                
                # Convert old format to new format if necessary
                if cache_data and not isinstance(next(iter(cache_data.values())), dict):
                    print("Converting cache from old format to new format...")
                    new_cache = {}
                    for entity, qid in cache_data.items():
                        new_cache[entity] = {
                            "hits": 1,
                            "ids": [qid] if qid is not None else []
                        }
                    return new_cache
                return cache_data
        return {}

    def _save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)
            
    def _prune_cache(self):
        """Prune the cache to maximum size by removing least frequently accessed entries."""
        if len(self.cache) <= self.max_cache_size:
            return
        
        # Sort entities by hit count (ascending)
        sorted_entities = sorted(
            [(entity, data["hits"]) for entity, data in self.cache.items()],
            key=itemgetter(1)
        )
        
        # Calculate how many to remove
        to_remove = len(self.cache) - self.max_cache_size
        entities_to_remove = [entity for entity, _ in sorted_entities[:to_remove]]
        
        # Remove the entities with the lowest hit counts
        for entity in entities_to_remove:
            del self.cache[entity]
            
        print(f"Pruned {to_remove} least used entries from cache. New size: {len(self.cache)} entries")

    def _disamb_entity(self, entity: str):
        """
        Return primary QID for entity if available, track all potential IDs.
        Returns None if no QID is found.
        """
        entity = entity.strip().strip("'\"")
        if entity.lower().startswith("and "):
            entity = entity[4:]
        self.total_entities += 1
        
        # Check if None
        if not entity or entity.lower() == "none":
            return None
        
        # If numeric (int or float), return as string
        try:
            # Try float first, then check if it's actually an int
            float_val = float(entity)
            if float_val.is_integer():
                return str(int(float_val))
            else:
                return str(float_val)
        except ValueError:
            pass
        
        # Check cache and update hit count
        if entity in self.cache:
            self.cache_hits += 1
            self.cache[entity]["hits"] += 1
            # Return the first ID if available, otherwise None
            return self.cache[entity]["ids"][0] if self.cache[entity]["ids"] else None
        
        # Call Wikidata
        url = (
            "https://www.wikidata.org/w/api.php?action=wbsearchentities"
            f"&search={entity}&language=en&format=json"
        )
        try:
            self.api_calls += 1
            data = requests.get(url).json()
            
            # Extract all IDs from the search results
            all_ids = [item["id"] for item in data.get("search", [])]
            
            # Create cache entry with hit count and all found IDs
            self.cache[entity] = {
                "hits": 1,
                "ids": all_ids
            }
            
            # Return the first ID if available, else None
            primary_qid = all_ids[0] if all_ids else None
            if primary_qid:
                self.successful_disambs += 1
            return primary_qid
        except Exception as e:
            # Create cache entry with hit but empty IDs
            self.cache[entity] = {
                "hits": 1,
                "ids": []
            }
            return None

    def _extract_entities_from_generated(self, raw_generated):
        """
        Extract entities from generated text, handling both regular and CoT formats.
        For CoT format, looks for <think>...</think> pattern and extracts the final answer.
        
        Args:
            raw_generated: The raw generated text from the model
            
        Returns:
            str: The extracted answer text with thinking tags removed
        """
        # Check if this is CoT format with <think> tags
        if "<think>" in raw_generated and "</think>" in raw_generated:
            # Extract the part after </think>
            try:
                answer_part = raw_generated.split("</think>", 1)[1].strip()
            except IndexError:
                answer_part = raw_generated  # Fallback if format is unexpected
        else:
            # Regular format (no CoT)
            answer_part = raw_generated
            
        return answer_part

    def disambiguate_csv(self, csv_file):
        """
        CSV columns: 
          SubjectEntityID, SubjectEntity, Relation, ObjectEntitiesID, ObjectEntities, generated
        """
        # Reset counters for this file
        self.total_entities = 0
        self.cache_hits = 0
        self.api_calls = 0
        self.successful_disambs = 0
        
        temp_file = csv_file + ".tmp"
        with open(csv_file, "r", encoding="utf-8") as fin, \
             open(temp_file, "w", newline="", encoding="utf-8") as fout:
            reader = csv.DictReader(fin)
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                raw_generated = row["generated"]
                # First remove any thinking tags using the extract method
                clean_output = self._extract_entities_from_generated(raw_generated)
                
                # Then split by comma to get individual entities
                entities = [e.strip() for e in clean_output.split(",") if e.strip()]
                
                disamb_ids = []
                disamb_names = []
                for entity in entities:
                    qid = self._disamb_entity(entity)
                    if qid:
                        disamb_ids.append(qid)
                        disamb_names.append(entity)

                row["ObjectEntitiesID"] = str(disamb_ids)
                row["ObjectEntities"] = str(disamb_names)
                writer.writerow(row)

        os.replace(temp_file, csv_file)
        
        # Prune the cache to maintain max size after processing the file
        self._prune_cache()
        self._save_cache()
        
        # Print disambiguation statistics
        print(f"\nDisambiguation Statistics for {os.path.basename(csv_file)}:")
        print(f"  Total entities processed: {self.total_entities}")
        print(f"  Cache hits: {self.cache_hits} ({self.cache_hits/self.total_entities:.1%})")
        print(f"  API calls: {self.api_calls}")
        print(f"  Successful disambiguations: {self.successful_disambs} ({self.successful_disambs/self.total_entities:.1%})")
        print(f"  Current cache size: {len(self.cache)} entries")
        
        # Report on cache efficiency
        if self.cache:
            avg_hits = sum(data["hits"] for data in self.cache.values()) / len(self.cache)
            max_hits = max(data["hits"] for data in self.cache.values())
            print(f"  Average cache hits per entity: {avg_hits:.2f}")
            print(f"  Maximum cache hits for an entity: {max_hits}")

    def run_evaluation(self, predictions_csv, ground_truth_csv):
        """
        Evaluate the disambiguated CSV against ground truth using evaluate.py.
        """
        results_df = evaluate(predictions_csv, ground_truth_csv)
        # Change output name from .json to results.json
        out_json = os.path.join(os.path.dirname(predictions_csv), "results.json")
        results_df.to_json(out_json, orient="index", indent=2)
        print(f"Evaluation results saved to {out_json}")