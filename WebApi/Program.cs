using Microsoft.Extensions.ML;
using Microsoft.OpenApi.Models;
using MentalHealthSentimentAnalysisAPI.Model;
using MentalHealthSentimentAnalysisAPI.WebApi.Services;
using System.Reflection;

namespace MentalHealthSentimentAnalysisAPI.WebApi;

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

        builder.Services.AddScoped<ISentimentAnalysisService, SentimentAnalysisService>(); // Register the service

        // Add MVC and Swagger
        builder.Services.AddControllers();
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen(c =>
        {
            c.SwaggerDoc("v1", new OpenApiInfo { Title = "Mental Health Sentiment Analysis API", Version = "v1" });

            // Include XML comments for API documentation
            var xmlFile = $"{Assembly.GetExecutingAssembly().GetName().Name}.xml";
            var xmlPath = Path.Combine(AppContext.BaseDirectory, xmlFile);
            c.IncludeXmlComments(xmlPath);
        });

        builder.Services
            .AddPredictionEnginePool<MentalHealthData, MentalHealthPrediction>()
            .FromFile(
                filePath: Path.Combine(
                AppContext.BaseDirectory, "..", "..", "..", "..", "Data", "MentalHealthModel.zip"),
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
