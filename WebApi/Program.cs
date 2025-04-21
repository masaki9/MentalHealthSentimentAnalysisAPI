using Microsoft.Extensions.ML;
using MentalHealthSentimentAnalysisAPI.Model;

namespace WebApi;

/// <summary>
/// The main entry point for the application.
/// </summary>
public class Program
{
    public static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);

        // Listen on the specified HTTP and HTTPS ports
        builder.WebHost.UseUrls("http://localhost:5000", "https://localhost:5001");

        // Add MVC and Swagger
        builder.Services.AddControllers();
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen();

        // Register the ML model
        builder.Services
            .AddPredictionEnginePool<MentalHealthData, MentalHealthPrediction>()
            .FromFile(
                filePath: Path.Combine(builder.Environment.ContentRootPath, "..", "Data", "MentalHealthModel.zip"),
                watchForChanges: true);

        var app = builder.Build();

        // Configure the HTTP request pipeline.
        if (app.Environment.IsDevelopment())
        {
            // Enable Swagger UI
            app.UseSwagger();
            app.UseSwaggerUI();
            app.UseDeveloperExceptionPage();
        }

        app.UseRouting();
        app.UseAuthorization();
        app.MapControllers();

        app.Run();
    }
}
