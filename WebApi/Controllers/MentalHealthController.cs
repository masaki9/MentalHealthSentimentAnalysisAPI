using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using MentalHealthSentimentAnalysisAPI.Model;

namespace WebApi.Controllers;

/// <summary>
/// Controller for analyzing mental health statements.
/// </summary>
[ApiController]
[Route("api/[controller]")]
public class MentalHealthController : ControllerBase
{
    private readonly PredictionEnginePool<MentalHealthData, MentalHealthPrediction> _engine;

    /// <summary>
    /// Constructor for MentalHealthController.
    /// </summary>
    /// <param name="engine">Prediction engine pool for mental health data.</param>
    public MentalHealthController(PredictionEnginePool<MentalHealthData, MentalHealthPrediction> engine)
    {
        _engine = engine;
    }

    /// <summary>
    /// Analyzes a mental health statement and predicts its status.
    /// </summary>
    /// <param name="input">The input data containing the statement.</param>
    /// <returns>An ActionResult containing the prediction result.</returns>
    [HttpPost("analyze")]
    public ActionResult Analyze([FromBody] MentalHealthData input)
    {
        if (string.IsNullOrWhiteSpace(input.Statement))
        {
            return BadRequest("Statement cannot be null or empty.");
        }

        var result = _engine.Predict(input);
        return Ok(new
        {
            statement = input.Statement,
            predictedStatus = result.Prediction,
            scores = result.Score
        });
    }
}
