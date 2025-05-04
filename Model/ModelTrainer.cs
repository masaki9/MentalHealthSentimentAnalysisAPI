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
    public TrainTestData LoadData(MLContext mlContext, string dataPath)
    {
        // Load the raw CSV data
        var rawData = mlContext.Data.LoadFromTextFile<MentalHealthData>(path: dataPath, separatorChar: ',', hasHeader: true, allowQuoting: true);

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
        var cleaned = cleaner.Fit(rawData).Transform(rawData);

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
        var estimator = mlContext.Transforms.Conversion.MapValueToKey("Label"); // Convert the label to a key type

        // Normalize the text to lowercase and remove punctuations
        var normalize = mlContext.Transforms.Text.NormalizeText(outputColumnName: "CleanText",
                                                                inputColumnName: nameof(MentalHealthData.Statement),
                                                                caseMode: TextNormalizingEstimator.CaseMode.Lower,
                                                                keepDiacritics: false,
                                                                keepPunctuations: false,
                                                                keepNumbers: true);

        // Tokenize the text into words
        var tokenize = mlContext.Transforms.Text.TokenizeIntoWords(outputColumnName: "Tokens", inputColumnName: "CleanText");

        // Remove stop-words
        IEstimator<ITransformer> stopWordsTransform;
        if (removeStopWords)
        {
            stopWordsTransform = mlContext.Transforms.Text.RemoveDefaultStopWords(outputColumnName: "TokensClean", inputColumnName: "Tokens");
        }
        else
        {
            // If stop-words are not to be removed, just copy the tokens
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
        var dataPath = Path.Combine(Environment.CurrentDirectory, "..", "Data", "MentalHealthData.csv");
        var splitDataView = LoadData(mlContext, dataPath);

        // Build featurizer pipeline
        var featurizer = BuildTextFeaturizerTfIdf(mlContext, ngramLength: 2, useAllLengths: true,
                                                maximumNgramsCount: 1000, removeStopWords: true);

        ITransformer? bestModel = null;
        var bestName = "";
        var bestMacro = 0.0;

        // Define the trainers to be evaluated
        var trainers = new (string name, IEstimator<ITransformer> trainer)[]
        {
            ("SDCA",  mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                        new SdcaMaximumEntropyMulticlassTrainer.Options {
                            LabelColumnName   = "Label",
                            FeatureColumnName = "Features"
                        })),
            ("LBFGS", mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                        new LbfgsMaximumEntropyMulticlassTrainer.Options {
                            LabelColumnName   = "Label",
                            FeatureColumnName = "Features"
                        })),
            ("OVA-LR", mlContext.MulticlassClassification.Trainers.OneVersusAll(
                        mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                            new LbfgsLogisticRegressionBinaryTrainer.Options {
                                LabelColumnName   = "Label",
                                FeatureColumnName = "Features"
                            })))
        };

        // Evaluate each trainer
        foreach (var (name, trainerEstimator) in trainers)
        {
            Console.WriteLine($"\n=== {name} ===");

            var fullPipeline = featurizer.AppendCacheCheckpoint(mlContext) // Caching to speed up training
                                         .Append(trainerEstimator)
                                         .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            var sw = Stopwatch.StartNew();
            var model = fullPipeline.Fit(splitDataView.TrainSet);
            sw.Stop();

            Console.WriteLine($"{name} train time: {sw.Elapsed}");

            var preds = model.Transform(splitDataView.TestSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(preds,
                                                                      labelColumnName: "Label",
                                                                      scoreColumnName: "Score",
                                                                      predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"{name}: MicroAcc={metrics.MicroAccuracy:F3}, " +
                              $"MacroAcc={metrics.MacroAccuracy:F3}, LogLoss={metrics.LogLoss:F2}");

            if (metrics.MicroAccuracy > bestMacro)
            {
                bestMacro = metrics.MacroAccuracy;
                bestModel = model;
                bestName = name;
            }
        }

        // Save the best model
        var modelPath = Path.Combine(Environment.CurrentDirectory, "..", "Data", "MentalHealthModel.zip");
        Console.WriteLine($"\nSaving best model '{bestName}' " + $"(MacroAcc={bestMacro:F3}) to {modelPath}");
        mlContext.Model.Save(bestModel!, splitDataView.TrainSet.Schema, modelPath);
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
