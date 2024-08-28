import sys
import asyncio
import json
import dataclasses
from app_migrator_analysis import MigrationSummarizer
import time

async def generate_migration_report(source_directory, access_key, output_file, gemini_version):
    # Example implementation

    print(f"[Gemini Migration Assistant v{gemini_version}] Analyzing project in directory: {source_directory}")

    print(f"[Gemini Migration Assistant] Detailed logs can be found in 'gemini_migration.log'") 

    summarizer = MigrationSummarizer(access_key, gemini_version)

    analysis_start_time = time.time()  # Start timing the analysis
    summaries = await summarizer.analyze_project(source_directory)
    analysis_end_time = time.time()
    analysis_execution_time = analysis_end_time - analysis_start_time

    print(f"[Gemini Migration Assistant] Migration analysis complete. Execution time: {analysis_execution_time:.2f} seconds. Saving detailed changes to 'app-migration-diff.json'")

    with open("app-migration-diff.json", "w") as outfile:
        json.dump([[dataclasses.asdict(x) for x in files] for files in summaries], outfile, sort_keys=True, indent=2)

    print("[Gemini Migration Assistant] Generating migration report...")

    report_start_time = time.time()  # Start timing the report generation
    await summarizer.summarize_report(summaries, output_file)
    report_end_time = time.time()
    report_execution_time = report_end_time - report_start_time

    print(f"[Gemini Migration Assistant] Migration report generated and saved to: {output_file}. Execution time: {report_execution_time:.2f} seconds")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python path_to_your_python_script.py <source_directory> <access_key> <output_file> [<gemini_version>]")
        sys.exit(1)
    
    source_directory = sys.argv[1]
    access_key = sys.argv[2]
    output_file = sys.argv[3]
    gemini_version = sys.argv[4] if len(sys.argv) > 4 else "gemini-1.5-flash-001"

    asyncio.run(generate_migration_report(source_directory, access_key, output_file, gemini_version))




