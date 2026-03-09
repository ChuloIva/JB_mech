  ---                                                                                                                                                                                               
  1. Attack Taxonomy / Clustering                                                                                                                                                                   
  Embed the descriptions and cluster to find attack "families":                                                                                                                                     
  # Use sentence-transformers to embed descriptions                                                                                                                                                 
  # Then UMAP + HDBSCAN to find natural clusters                                                                                                                                                    
  # Visualize with plotly/matplotlib                                                                                                                                                                
  Goal: Discover attack patterns that aren't captured by tags alone                                                                                                                                 
                                                                                                                                                                                                    
  ---                                                                                                                                                                                               
  2. Model Vulnerability Matrix                                                                                                                                                                     
  Cross-tabulate models × attack tags:                                                                                                                                                              
  # Pivot table: rows=models, cols=tags, values=count                                                                                                                                               
  # Heatmap showing which models are weak to what                                                                                                                                                   
  Goal: See if certain model families have blind spots (e.g., "Claude weak to X, GPT weak to Y")                                                                                                    
                                                                                                                                                                                                    
  ---                                                                                                                                                                                               
  3. Temporal Trend Analysis                                                                                                                                                                        
  Track attack sophistication over time using paperDate:                                                                                                                                            
  # Group by month/quarter                                                                                                                                                                          
  # See which attack types are increasing                                                                                                                                                           
  # Correlate with model releases                                                                                                                                                                   
  Goal: Predict emerging threat vectors                                                                                                                                                             
                                                                                                                                                                                                    
  ---                                                                                                                                                                                               
  4. Extract Attack Templates                                                                                                                                                                       
  Parse descriptions for actionable patterns:                                                                                                                                                       
  # Use an LLM to extract:                                                                                                                                                                          
  # - Attack setup steps                                                                                                                                                                            
  # - Example prompts/payloads                                                                                                                                                                      
  # - Bypass techniques                                                                                                                                                                             
  # Build a structured red-team playbook                                                                                                                                                            
  Goal: Create a usable jailbreak/injection test suite                                                                                                                                              
                                                                                                                                                                                                    
  ---                                                                                                                                                                                               
  5. Paper Network Analysis                                                                                                                                                                         
  Scrape arXiv links for citations, build a graph:                                                                                                                                                  
  # Which papers cite which                                                                                                                                                                         
  # Find foundational attack research                                                                                                                                                               
  # Identify research groups/clusters                                                                                                                                                               
  Goal: Understand the research landscape   