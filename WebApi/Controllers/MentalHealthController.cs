using Microsoft.AspNetCore.Mvc;
using MentalHealthSentimentAnalysisAPI.WebApi.Services;

namespace MentalHealthSentimentAnalysisAPI.WebApi.Controllers;

/// <summary>
/// Controller for analyzing mental health statements.
/// </summary>
[ApiController]
[Route("api/[controller]")]
public class MentalHealthController : ControllerBase
{
    private readonly ISentimentAnalysisService _sentimentAnalysisService;

    public MentalHealthController(ISentimentAnalysisService service)
    {
        _sentimentAnalysisService = service;
    }

    /// <summary>
    /// Analyzes a mental health statement and returns the predicted status and scores.
    /// </summary>
    /// <param name="statement">Statement to analyze.</param>
    /// <returns>Predicted mental health status and scores.</returns>
    /// <response code="200">Returns the predicted mental health status and scores.</response>
    /// <response code="400">Bad request if the statement is null or empty.</response>
    [HttpPost("analyze")]
    public ActionResult Analyze([FromBody] string statement)
    {
        if (string.IsNullOrWhiteSpace(statement))
        {
            return BadRequest("Statement cannot be null or empty.");
        }

        var result = _sentimentAnalysisService.Analyze(statement);
        return Ok(new
        {
            statement,
            predictedStatus = result.Prediction,
            scores = result.Score
        });
    }
}
