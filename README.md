# Predicting Barbell Lift Quality

Course project by Mrunmayi A Bhave. The report source and compiled HTML are included, along with the quiz predictions.

## Reproduce the analysis

1. Install R and the two packages used by the project: `install.packages(c("randomForest", "knitr"))`.
2. Download the course data into a `data` folder in this directory:

   ```r
   dir.create("data", showWarnings = FALSE)
   download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "data/pml-training.csv", mode = "wb")
   download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "data/pml-testing.csv", mode = "wb")
   ```

3. Run `Rscript analysis.R` to perform stratified five-fold cross-validation, fit the final model, and create quiz predictions.
4. Run `Rscript -e 'knitr::knit("model_report.Rmd", output="model_report.html")'` to rebuild the HTML report.

The report estimates out-of-sample error from pooled out-of-fold predictions. It describes the model, validation results, and limitations. `quiz_predictions.txt` lists the 20 predictions in quiz order; `quiz_predictions.csv` includes row numbers.

## Data source

Groupware at PUC-Rio, Weight Lifting Exercise Dataset: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/harsvg>.

