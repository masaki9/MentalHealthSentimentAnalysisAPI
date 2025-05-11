using Microsoft.Extensions.ML;
using MentalHealthSentimentAnalysisAPI.Model;

namespace MentalHealthSentimentAnalysisAPI.WebApi.Services;

/// <summary>
/// Service for sentiment analysis.
/// </summary>
public class SentimentAnalysisService : ISentimentAnalysisService
{
    private readonly PredictionEnginePool<MentalHealthData, MentalHealthPrediction> _engine;

    /// <summary>
    /// Constructor for SentimentAnalysisService.
    /// </summary>
    public SentimentAnalysisService(PredictionEnginePool<MentalHealthData, MentalHealthPrediction> engine)
    {
        _engine = engine;
    }

    /// <summary>
    /// Predicts the mental health status and scores based on the given statement.
    /// </summary>
    /// <param name="statement">Statement to analyze.</param>
    /// <returns>Prediction of the mental health status.</returns>
    /// <exception cref="ArgumentException"></exception>
    public MentalHealthPrediction Analyze(string statement)
    {
        var input = new MentalHealthData { Statement = statement };
        return _engine.Predict(input);
    }
}
