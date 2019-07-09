# Topic-analysis(NLP) to access the level of damage during a car crash.
Data containing 2,375 complaints about specific GMC vehicles submitted to the National Highway Safety and Traffic Administration (NHTSA).

The complaints were under column description along with several other variables such as make, Model, year, Mileage of the vehicle. 
Conducted a text classification analysis on this data

Parts of speech tagging, Stop_words removal, stemming( root words collapsing) for building the term/doc matrix.

Used TF-IDF weights for Term/Doc Matrix

A total of 8 Clusters of complaints topics were formed.

probability of a observation falling into one of these topics Clusters(T1 to T8) along with make, model, year, mileage were used to predict the level of damage and hence the level of emergency.
