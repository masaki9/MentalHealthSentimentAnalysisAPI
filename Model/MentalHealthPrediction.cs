using Microsoft.ML.Data;

namespace MentalHealthSentimentAnalysisAPI.Model;

public class MentalHealthPrediction
{
    [ColumnName("PredictedLabel")]
    public string? Prediction { get; set; }

    [ColumnName("Score")]
    public float[]? Score { get; set; } // Array of scores for each class
}
