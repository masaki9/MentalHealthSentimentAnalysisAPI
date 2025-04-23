

using Microsoft.ML.Data;

namespace MentalHealthSentimentAnalysisAPI.Model;

public class MentalHealthData
{
    [LoadColumn(0)]
    public int Id { get; set; }

    [LoadColumn(1)]
    public string? Statement { get; set; }

    [LoadColumn(2), ColumnName("Label")]
    public string? Status { get; set; }
}
