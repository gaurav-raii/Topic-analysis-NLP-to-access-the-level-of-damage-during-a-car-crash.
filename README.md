# Topic-analysis(NLP) to access the level of damage during a car crash.

Developed a model to assess the damage to a vehicle during an accident using the text data from complaints submitted to National Highway safety and Traffic Administration( NHSTA). 
The data contained 2,375 complaints about specific GMC vehicles submitted to the NHTSA. The complaints were under column description along with several other variables such as make, Model, year, Mileage of the vehicle.
Conducted a text classification analysis on this data ,using POS tagging , Stop_words removal, stemming for building the term/doc matrix. Used TF-IDF weights for Term/Doc Matrix. 8 Clusters of complaints were formed. Probability of an observation falling into one of these topics Clusters(T1 to T8) along with make, model, year, mileage were used to classify the level of damage into low, medium and high. This type of models can be used to automatically prioritize and route the resources based on submitted information in emergency scenarios such as a wild fires and natural disasters where saving even seconds in taking actions can make all the difference.

Tools Used: Python
Packages: NLKT, Pandas.
