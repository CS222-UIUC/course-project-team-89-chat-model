using Microsoft.AspNetCore.Mvc;

namespace webapi.Controllers;

[ApiController]
[Route("[controller]")]
public class CodeGeneratorController : ControllerBase
{
    private static readonly string[] Randoms = new[]
    {
        "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"
    };

    private readonly ILogger<CodeGeneratorController> _logger;

    public CodeGeneratorController(ILogger<CodeGeneratorController> logger)
    {
        _logger = logger;
    }

    [HttpGet(Name = "GetCode")]
    public IEnumerable<CodeGenerator> Get()
    {
        return Enumerable.Range(1, 5).Select(index => new CodeGenerator
        {
            RandomInt = Random.Shared.Next(0, 100)
            RandomStr = Randoms[Random.Shared.Next(Randoms.Length)]
        })
        .ToArray();
    }
}
