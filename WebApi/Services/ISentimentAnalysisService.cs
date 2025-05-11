using MentalHealthSentimentAnalysisAPI.Model;

namespace MentalHealthSentimentAnalysisAPI.WebApi.Services;

/// <summary>
/// Interface for sentiment analysis service.
/// </summary>
public interface ISentimentAnalysisService
{
    /// <summary>
    /// Predicts the mental health status based on the given statement.
    /// </summary>
    /// <param name="statement">Statement to analyze.</param>
    /// <returns>Prediction of the mental health status.</returns>
    MentalHealthPrediction Analyze(string statement);
}
