  # -v: Shows turn structure, truncated messages, evaluation verdict                                                                   
  python benchmark.py -d harmbench -n 2 -t openai/gpt-4o-mini -o arcee-ai/trinity-large-preview:free -v                                
                                                                                                                                       
  # -vv: Shows full messages, reasoning, what worked/failed, strategy suggestions                                                      
  python benchmark.py -d harmbench -n 2 -t openai/gpt-4o-mini -o arcee-ai/trinity-large-preview:free -vv 