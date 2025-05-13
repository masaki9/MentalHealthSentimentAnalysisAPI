using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace MentalHealthSentimentAnalysisAPI.Model;

/// <summary>
/// Class to hold utility methods for model training.
/// </summary>
public static class ModelTrainerUtils
{
    /// <summary>
    /// Adds a sweep of varying L2 regularization strengths to the specified estimator configurations.
    /// </summary>
    /// <param name="prefix"></param>
    /// <param name="l2Grid"></param>
    /// <param name="factory"></param>
    /// <param name="estimatorConfigs"></param>
    public static void AddL2Sweep(string prefix, IEnumerable<float> l2Grid, Func<float?, IEstimator<ITransformer>> factory, ICollection<(string, IEstimator<ITransformer>)> estimatorConfigs)
    {
        foreach (var l2 in l2Grid)
        {
            estimatorConfigs.Add(($"{prefix}-L2={l2:g}", factory(l2)));
        }

        estimatorConfigs.Add(($"{prefix} (No Regularization)", factory(null)));
    }

    /// <summary>
    /// Builds an SdcaMaximumEntropy trainer with the specified L2 regularization strength.
    /// </summary>
    /// <param name="mlContext"></param>
    /// <param name="l2"></param>
    public static IEstimator<ITransformer> BuildSdca(MLContext mlContext, float? l2)
    {
        var options = new SdcaMaximumEntropyMulticlassTrainer.Options
        {
            LabelColumnName = "LabelKey",
            FeatureColumnName = "Features"
        };

        if (l2.HasValue)
        {
            options.L2Regularization = l2.Value;
        }

        return mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(options);
    }

    /// <summary>
    /// Builds an LbfgsMaximumEntropy trainer with the specified L2 regularization strength.
    /// </summary>
    /// <param name="mlContext"></param>
    /// <param name="l2"></param>
    public static IEstimator<ITransformer> BuildLbfgs(MLContext mlContext, float? l2)
    {
        var options = new LbfgsMaximumEntropyMulticlassTrainer.Options
        {
            LabelColumnName = "LabelKey",
            FeatureColumnName = "Features"
        };

        if (l2.HasValue)
        {
            options.L2Regularization = l2.Value;
        }

        return mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(options);
    }

    /// <summary>
    /// Builds an One-Versus-All + SDCA trainer with the specified L2 regularization strength.
    /// </summary>
    /// <param name="mlContext"></param>
    /// <param name="l2"></param>
    public static IEstimator<ITransformer> BuildOvaLbfgs(MLContext mlContext, float? l2)
    {
        var options = new SdcaLogisticRegressionBinaryTrainer.Options
        {
            LabelColumnName = "LabelKey",
            FeatureColumnName = "Features"
        };

        if (l2.HasValue)
        {
            options.L2Regularization = l2.Value;
        }

        var binaryTrainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(options);
        return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer, labelColumnName: "LabelKey");
    }

    /// <summary>
    /// Builds an One-Versus-All + LBFGS trainer with the specified L2 regularization strength.
    /// </summary>
    /// <param name="mlContext"></param>
    /// <param name="l2"></param>
    public static IEstimator<ITransformer> BuildOvaSdca(MLContext mlContext, float? l2)
    {
        var options = new LbfgsLogisticRegressionBinaryTrainer.Options
        {
            LabelColumnName = "LabelKey",
            FeatureColumnName = "Features"
        };

        if (l2.HasValue)
        {
            options.L2Regularization = l2.Value;
        }

        var binaryTrainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(options);
        return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer, labelColumnName: "LabelKey");
    }
}
