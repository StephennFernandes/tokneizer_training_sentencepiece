import datasets 
from src.make_datasets import make_sentence_files


assamese_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/ASSAMESE")
bengali_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/BENGALI")
bodo_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/BODO")
bhisnupuriya_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/BHISNUPURIYA")
divehi_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/DIVEHI")
dogri_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/DOGRI")
english_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/ENGLISH")
gujarati_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/GUJARATI")
hindi_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/HINDI")
kannada_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/KANNADA")
konkani_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/KONKANI")
maithili_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/MAITHILI")
malayalam_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/MALAYALAM")
manipuri_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/MANIPURI")
marathi_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/MARATHI")
nepali_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/NEPALI")
odia_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/ODIA")
panjabi_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/PANJABI")
sanskrit_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/SANSKRIT")
tamil_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/TAMIL")
telugu_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/TELUGU")
urdu_dataset = datasets.load_from_disk("~/gcs_bucket/COMBINED_CORPUS/URDU")

combined_dataset = datasets.concatente_datasets(assamese_dataset["train"], 
                                                assamese_dataset["validation"], 
                                                
                                                bengail_dataset["train"], 
                                                bengali_dataset["validation"], 
                                                
                                                bodo_dataset["train"], 
                                                bodo_dataset["validation"], 

                                                bhisnupuriya_dataset["train"], 
                                                bhisnupuriya_dataset["validation"], 
                                                
                                                divehi_dataset["train"], 
                                                divehi_dataset["validation"],

                                                dogri_dataset["train"],
                                                dogri_dataset["validation"], 

                                                english_dataset["train"], 
                                                english_dataset["validation"], 

                                                gujarati_dataset["train"], 
                                                gujarati_dataset["validation"], 

                                                hindi_dataset["train"], 
                                                hindi_dataset["validation"], 
                                                
                                                kannada_dataset["train"], 
                                                kannada_dataset["validation"], 

                                                kashmiri_dataset["train"], 
                                                kashmiri_dataset["validation"], 

                                                konkani_dataset["train"], 
                                                konkani_dataset["validation"], 

                                                maithili_dataset["train"], 
                                                maithili_dataset["validation"], 

                                                manipuri_dataset["train"], 
                                                manipuri_dataset["validation"], 

                                                malayalam_dataset["train"], 
                                                malayalam_dataset["validation"],

                                                marathi_dataset["train"], 
                                                marathi_dataset["validation"], 

                                                nepali_dataset["train"], 
                                                nepali_dataset["validation"], 

                                                odia_dataset["train"], 
                                                odia_dataset["validation"],

                                                panjabi_dataset["train"], 
                                                panjabi_dataset["validation"], 

                                                sanskrit_dataset["train"], 
                                                sanskrit_dataset["validation"], 

                                                tamil_dataset["train"], 
                                                tamil_dataset["validtion"],

                                                telugu_dataset["train"], 
                                                telugu_dataset["validation"], 

                                                urdu_dataset["train"], 
                                                urdu_dataset["validation"])
print(combined_dataset)

make_sentence_files(combined_dataset["train"])