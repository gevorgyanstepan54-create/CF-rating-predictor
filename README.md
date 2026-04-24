# CF-rating-predictor
This is a ML project, that estimates the rating change in cf, for a given division,precentile performance and current rating,and predicts the new rating, after the contest. 
The model supports the following contest types: div1, div2_solo (where 1900–2100 rated participants are allowed), div2_parallel (where 1900–2100 rated participants are not allowed), div1_div2, and div3.
The model has 3 functions - predict_rating_change,estimate_performance_rating and predict_all. the first 2 do what is said in their names, the last one does both. 
The inputs of functions are a string (either "div1","div2_solo","div2_parallel","div3" or "div1_div2". div2_solo corresponds to a contest, that does not have a div1 contest going at the same time,div2_parallel corresponds to a contest, that has a div1 contest going at the same time, div1_div2 corresponds to div1+div2 types of contests), the rating begor the contest, and the precentile of the participant (place/number of participants).
It's important to note, that the models were trained on data of participants with 1000+ rating, so it will work good only for those participants.
