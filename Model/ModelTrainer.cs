using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using System.Diagnostics;
using System.Text.RegularExpressions;

namespace MentalHealthSentimentAnalysisAPI.Model;

/// <summary>
/// Class to train, evaluate, and make predictions using a machine learning model for mental health sentiment analysis.
/// </summary>
public class ModelTrainer
{
    /// <summary>
    /// Load data, clean it, and split it into training and testing sets.
    /// </summary>
    /// <param name="mlContext">The MLContext instance.</param>
    /// <param name="dataPath">The path to the data file.</param>
    /// <returns>A TrainTestData object containing the training and testing data.</returns>
    public TrainTestData LoadAndCleanData(MLContext mlContext, string dataPath)
    {
        // Load the raw CSV data
        var rawData = mlContext.Data.LoadFromTextFile<MentalHealthData>(path: dataPath, separatorChar: ',', hasHeader: true, allowQuoting: true);

        // Filter out null/empty statements
        var nonEmpty = mlContext.Data.CreateEnumerable<MentalHealthData>(rawData, reuseRowObject: false)
                                     .Where(r => !string.IsNullOrWhiteSpace(r.Statement));

        // Convert back into IDataView
        var filteredData = mlContext.Data.LoadFromEnumerable(nonEmpty);

        // Define a custom mapping action to clean the data
        Action<MentalHealthData, MentalHealthData> cleanData = (input, output) =>
        {
            output.Status = input.Status; // Copy over the label

            var s = (input.Statement ?? "").ToLowerInvariant(); // Normalize to lowercase

            s = Regex.Replace(s, @"\[.*?\]", "", RegexOptions.Singleline); // Remove bracketed text
            s = Regex.Replace(s, @"https?://\S+|www\.\S+", "", RegexOptions.IgnoreCase); // Remove URLs
            s = Regex.Replace(s, @"<.*?>", "", RegexOptions.Singleline); // Remove HTML tags
            s = Regex.Replace(s, @"[\p{P}\p{S}]", "", RegexOptions.None); // Remove punctuations
            s = Regex.Replace(s, @"\w*\d\w*", "", RegexOptions.None); // Remove digits
            s = s.Replace("\n", " "); // Replace newlines with spaces

            output.Statement = s;
        };

        // Build and apply the cleaner
        var cleaner = mlContext.Transforms.CustomMapping(cleanData, contractName: "CleanData");
        var cleaned = cleaner.Fit(rawData).Transform(filteredData);

        // Split the data into training and testing sets (80% training, 20% testing)
        return mlContext.Data.TrainTestSplit(cleaned, testFraction: 0.2, seed: 42);
    }

    /// <summary>
    /// Build the text featurization part of the pipeline.
    /// </summary>
    /// <param name="mlContext">The MLContext instance.</param>
    /// <param name="ngramLength">The length of the n-grams to be used.</param>
    /// <param name="useAllLengths">Whether to store all n-gram lengths up to ngramLength, or only ngramLength.</param>
    /// <param name="maximumNgramsCount">>The maximum number of n-grams to be used.</param>
    /// <param name="removeStopWords">Whether to remove stop words.</param>
    /// <returns>A text featurization estimator.</returns>
    public IEstimator<ITransformer> BuildTextFeaturizerTfIdf(MLContext mlContext, int ngramLength, bool useAllLengths, int maximumNgramsCount, bool removeStopWords)
    {
        // Convert the label to a key
        var estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey",
                                                                      inputColumnName: "Label",
                                                                      addKeyValueAnnotationsAsText: true);

        // Normalize the text to lowercase and remove punctuations
        var normalize = mlContext.Transforms.Text.NormalizeText(outputColumnName: "CleanText",
                                                                inputColumnName: nameof(MentalHealthData.Statement),
                                                                caseMode: TextNormalizingEstimator.CaseMode.Lower,
                                                                keepDiacritics: false,
                                                                keepPunctuations: false,
                                                                keepNumbers: true);

        // Tokenize the text into words
        var tokenize = mlContext.Transforms.Text.TokenizeIntoWords(outputColumnName: "Tokens", inputColumnName: "CleanText");

        // Remove stopwords
        IEstimator<ITransformer> stopWordsTransform;
        if (removeStopWords)
        {
            stopWordsTransform = mlContext.Transforms.Text.RemoveDefaultStopWords(outputColumnName: "TokensClean", inputColumnName: "Tokens");
        }
        else
        {
            // If stopwords are not to be removed, just copy the tokens
            stopWordsTransform = mlContext.Transforms.CopyColumns(outputColumnName: "TokensClean", inputColumnName: "Tokens");
        }

        // Map the tokens to keys
        var mapToKeys = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "TokensKey", inputColumnName: "TokensClean");

        // Create n-grams from the tokens
        var ngrams = mlContext.Transforms.Text.ProduceNgrams(outputColumnName: "Features",
                                                             inputColumnName: "TokensKey",
                                                             ngramLength: ngramLength,
                                                             useAllLengths: useAllLengths,
                                                             maximumNgramsCount: maximumNgramsCount,
                                                             weighting: NgramExtractingEstimator.WeightingCriteria.TfIdf);

        return estimator.Append(normalize).Append(tokenize).Append(stopWordsTransform).Append(mapToKeys).Append(ngrams);
    }

    /// <summary>
    /// Run the model trainer.
    /// </summary>
    /// <param name="args">Command line arguments.</param>
    /// <returns>None</returns>
    public void Run(string[] args)
    {
        var mlContext = new MLContext();
        var projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
        var dataPath = Path.Combine(projectRoot, "Data", "MentalHealthData.csv");
        var modelPath = Path.Combine(projectRoot, "Data", "MentalHealthModel.zip");
        var splitDataView = LoadAndCleanData(mlContext, dataPath);

        // Build text featurizer pipeline
        var featurizer = BuildTextFeaturizerTfIdf(mlContext, ngramLength: 2, useAllLengths: true,
                                                maximumNgramsCount: 50000, removeStopWords: true);

        IEstimator<ITransformer>? bestTrainer = null;
        var bestName = "";
        var bestMacro = 0.0;

        float[] l2Regs = new[] { 1e-2f, 1e-3f, 1e-4f, 1e-5f, 1e-6f, 1e-7f };
        var trainerConfigs = new List<(string name, IEstimator<ITransformer> est)>();

        ModelTrainerUtils.AddL2Sweep("SDCA", l2Regs, l2 => ModelTrainerUtils.BuildSdca(mlContext, l2), trainerConfigs);
        ModelTrainerUtils.AddL2Sweep("LBFGS", l2Regs, l2 => ModelTrainerUtils.BuildLbfgs(mlContext, l2), trainerConfigs);
        ModelTrainerUtils.AddL2Sweep("OVA+SDCA", l2Regs, l2 => ModelTrainerUtils.BuildOvaSdca(mlContext, l2), trainerConfigs);
        ModelTrainerUtils.AddL2Sweep("OVA+LBFGS", l2Regs, l2 => ModelTrainerUtils.BuildOvaLbfgs(mlContext, l2), trainerConfigs);

        var trainedTextFeatures = featurizer.Fit(splitDataView.TrainSet);
        var transformedTrainSet = trainedTextFeatures.Transform(splitDataView.TrainSet);
        var cachedTrainSet = mlContext.Data.Cache(transformedTrainSet);

        // Evaluate each trainer
        foreach (var (name, trainerEstimator) in trainerConfigs)
        {
            Console.WriteLine($"\n=== {name} ===");
            var trainingPipeline = trainerEstimator.Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            // Run 5-fold CV to evaluate the trainer on the cached training set
            var cvResults = mlContext.MulticlassClassification.CrossValidate(
                data: cachedTrainSet,
                estimator: featurizer.AppendCacheCheckpoint(mlContext).Append(trainingPipeline),
                numberOfFolds: 5,
                labelColumnName: "LabelKey"
            );

            var avgMacro = cvResults.Average(r => r.Metrics.MacroAccuracy);
            var avgMicro = cvResults.Average(r => r.Metrics.MicroAccuracy);
            var avgLog = cvResults.Average(r => r.Metrics.LogLoss);
            Console.WriteLine($"{name} (5-fold CV) → AvgMicro={avgMicro:F3}, AvgMacro={avgMacro:F3}, AvgLogLoss={avgLog:F2}");

            if (avgMacro > bestMacro)
            {
                bestMacro = avgMacro;
                bestName = name;
                bestTrainer = trainerEstimator;
            }
        }

        var dropCols = mlContext.Transforms.DropColumns("CleanText", "Tokens", "TokensClean", "TokensKey");

        // Build the final pipeline with the best trainer
        var finalPipeline = featurizer.AppendCacheCheckpoint(mlContext)
                                      .Append(bestTrainer!)
                                      .Append(mlContext.Transforms.Conversion
                                      .MapKeyToValue("PredictedLabel", "PredictedLabel"))
                                      .Append(dropCols);

        // Fit the best model on the entire training set
        Console.WriteLine($"\nRetraining best model [{bestName}] on the entire training set...");
        var bestModel = finalPipeline.Fit(splitDataView.TrainSet);

        // Evalute the best model on the test set
        var preds = bestModel.Transform(splitDataView.TestSet);
        var metrics = mlContext.MulticlassClassification.Evaluate(preds,
                                                                  labelColumnName: "LabelKey",
                                                                  scoreColumnName: "Score",
                                                                  predictedLabelColumnName: "PredictedLabel");

        Console.WriteLine($"Eval on Test Set: MicroAcc={metrics.MicroAccuracy:F3}, MacroAcc={metrics.MacroAccuracy:F3}, LogLoss={metrics.LogLoss:F2}");

        // Save the best model found during CV
        mlContext.Model.Save(bestModel, splitDataView.TrainSet.Schema, modelPath);
        Console.WriteLine($"Saved best model [{bestName}] to {modelPath}");
    }

    /// <summary>
    /// Entry point for the application.
    /// </summary>
    /// <param name="args">Command line arguments.</param>
    /// <returns>None</returns>
    public static void Main(string[] args)
    {
        var trainer = new ModelTrainer();
        trainer.Run(args);
    }
}
