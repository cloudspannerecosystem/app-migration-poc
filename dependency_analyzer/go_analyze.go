// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package main provides a tool to analyze Go projects and generate a dependency graph between files.
//
// This tool uses the Go packages API to identify dependencies between files in a project by analyzing
// symbol usage. It outputs a dependency graph in DOT format, which can be visualized using graph tools
// like Graphviz.
//
// Example Usage:
//
// Suppose you have a Go project in the directory `/my-project`:
//
// ```
// go run main.go -path /my-project
// ```
//
// This will analyze the project and generate a `dependency_graph.dot` file in the current directory.
// You can visualize the DOT file using a tool like Graphviz:
//
// ```
// dot -Tpng dependency_graph.dot -o graph.png
// ```
//
// The resulting PNG will show the dependency relationships between files in your project.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"golang.org/x/tools/go/packages"
)

func main() {
	// Define a command-line flag for the project path
	projectPath := flag.String("path", "./", "Path to the project to analyze")
	flag.Parse()

	// Configure the packages.Load function
	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedSyntax | packages.NeedTypes | packages.NeedTypesInfo,
		Dir:  (*projectPath),
	}

	project_base_path := *projectPath

	// Load the packages
	pkgs, err := packages.Load(cfg, "./...")
	if err != nil {
		log.Fatalf("Error loading packages: %v", err)
	}

	// Dependency graph: key = file, value = list of files it depends on
	dependencyGraph := make(map[string]map[string]struct{})

	// Iterate through all packages and process their files
	for _, pkg := range pkgs {
		if pkg.TypesInfo == nil {
			continue
		}

		// Process symbol usages (functions, variables, structs)
		for ident, obj := range pkg.TypesInfo.Uses {
			if obj != nil && obj.Pos().IsValid() {
				useFile := pkg.Fset.Position(ident.Pos()).Filename

				// Only process files inside the project directory
				if strings.HasPrefix(useFile, project_base_path) {
					// Get the file where the symbol is defined
					defFile := pkg.Fset.Position(obj.Pos()).Filename

					// Only add if the file is inside the project directory and avoid redundant edges
					if strings.HasPrefix(defFile, project_base_path) && useFile != defFile {
						// Initialize the map for the useFile if not present
						if _, ok := dependencyGraph[useFile]; !ok {
							dependencyGraph[useFile] = make(map[string]struct{})
						}

						// Add the dependency edge only if it doesn't already exist
						if _, exists := dependencyGraph[useFile][defFile]; !exists {
							dependencyGraph[useFile][defFile] = struct{}{}
						}
					}
				}
			}
		}
	}

	// Write the dependency graph to a DOT file
	dotFile, err := os.Create("dependency_graph.dot")
	if err != nil {
		log.Fatalf("Error creating DOT file: %v", err)
	}
	defer func() {
		if cerr := dotFile.Close(); cerr != nil {
			log.Printf("Error closing DOT file: %v", cerr)
		}
	}()

	// Write the DOT header
	if _, err := dotFile.WriteString("digraph G {\n"); err != nil {
		log.Fatalf("Error writing to DOT file: %v", err)
	}

	// Write the dependency graph edges
	for file, dependencies := range dependencyGraph {
		for dep := range dependencies {
			if _, err := dotFile.WriteString(fmt.Sprintf("\t\"%s\" -> \"%s\";\n", strings.TrimPrefix(file, project_base_path), strings.TrimPrefix(dep, project_base_path))); err != nil {
				log.Fatalf("Error writing edge to DOT file: %v", err)
			}
		}
	}

	// Write the DOT footer
	if _, err := dotFile.WriteString("}\n"); err != nil {
		log.Fatalf("Error writing DOT file footer: %v", err)
	}

	fmt.Println("Dependency graph has been written to dependency_graph.dot")
}
