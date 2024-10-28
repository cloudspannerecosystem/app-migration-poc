import asyncio
import dataclasses
import json
import sys
import time

from app_migrator_analysis import MigrationSummarizer


async def generate_migration_report(
    source_directory,
    mysql_schema_file,
    spanner_schema_file,
    access_key,
    output_file,
    gemini_version,
):
    # Example implementation

    print(
        f"[Gemini Migration Assistant v{gemini_version}] Analyzing project in directory: {source_directory}"
    )

    print(
        f"[Gemini Migration Assistant] Detailed logs can be found in 'gemini_migration.log'"
    )

    summarizer = MigrationSummarizer(access_key, gemini_version)

    analysis_start_time = time.time()  # Start timing the analysis
    summaries, files_metadata = await summarizer.analyze_project(
        source_directory, mysql_schema_file, spanner_schema_file
    )
    analysis_end_time = time.time()
    analysis_execution_time = analysis_end_time - analysis_start_time

    print(
        f"[Gemini Migration Assistant] Migration analysis complete. Execution time: {analysis_execution_time:.2f} seconds. Saving detailed changes to 'app-migration-diff.json'"
    )

    with open("app-migration-diff.json", "w") as outfile:
        json.dump(
            [[dataclasses.asdict(x) for x in files] for files in summaries],
            outfile,
            sort_keys=True,
            indent=2,
        )

    print("[Gemini Migration Assistant] Generating migration report...")

    report_start_time = time.time()  # Start timing the report generation
    await summarizer.summarize_report(summaries, files_metadata, output_file)
    report_end_time = time.time()
    report_execution_time = report_end_time - report_start_time

    print(
        f"[Gemini Migration Assistant] Migration report generated and saved to: {output_file}. Execution time: {report_execution_time:.2f} seconds"
    )


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print(
            "Usage: python path_to_your_python_script.py <source_directory> <mysql_schema_file> <spanner_schema_file> <access_key> <output_file> [<gemini_version>]"
        )
        sys.exit(1)

    source_directory = sys.argv[1]
    mysql_schema_file = sys.argv[2]
    spanner_schema_file = sys.argv[3]
    access_key = sys.argv[4]
    output_file = sys.argv[5]
    gemini_version = sys.argv[6] if len(sys.argv) >= 6 else "gemini-1.5-flash-001"

    asyncio.run(
        generate_migration_report(
            source_directory,
            mysql_schema_file,
            spanner_schema_file,
            access_key,
            output_file,
            gemini_version,
        )
    )
