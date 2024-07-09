# Copyright 2024 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tree_sitter # type: ignore
import os
import tree_sitter_java as tsjava # type: ignore
import networkx as nx # type: ignore

class JavaAnalyzer:
    JAVA_LANGUAGE = tree_sitter.Language(tsjava.language())
    parser = tree_sitter.Parser()
    parser.set_language(JAVA_LANGUAGE)

    def find_public_classes(self, node):
        """
        Recursively searches for public classes or interfaces within the given syntax tree node.

        Args:
            node (tree_sitter.Node): The root node to start searching from.

        Returns:
            list: A list of public class or interface names found in the syntax tree.
        """
        if node is None:
            return []

        public_classes = []

        # Check if the node represents a class or interface declaration
        if node.type in ['class_declaration', 'interface_declaration']:
            has_public_modifier = False

            # Check for the 'public' modifier
            for child in node.children:
                if child.type == 'modifiers':
                    has_public_modifier = any(
                        modifier.type == 'public' for modifier in child.children
                    )
                    break

            if has_public_modifier:
                class_name_node = node.child_by_field_name('name')
                if class_name_node:
                    public_classes.append(class_name_node.text.decode('utf-8'))

        # Recursively search child nodes
        for child in node.children:
            public_classes.extend(self.find_public_classes(child))

        return public_classes

    def find_public_classes_in_file(self, file_path):
        """
        Parses a Java file and extracts all public class or interface names.

        Args:
            file_path (str): The path to the Java file.

        Returns:
            list: A list of public class or interface names found in the file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                source_code = file.read()
                tree = self.parser.parse(bytes(source_code, "utf8"))
                root_node = tree.root_node

                return self.find_public_classes(root_node)
        except (FileNotFoundError, UnicodeDecodeError) as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def find_public_classes_in_project(self, project_dir):
        """
        Recursively searches for Java files in a project directory and extracts public classes or interfaces.

        Args:
            project_dir (str): The root directory of the Java project.

        Returns:
            tuple: Two dictionaries:
                - file_to_class (dict): Maps file paths to lists of public class or interface names.
                - class_to_file (dict): Maps class or interface names to their respective file paths.
        """
        file_to_class = {}
        class_to_file = {}

        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    public_classes = self.find_public_classes_in_file(file_path)
                    if public_classes:
                        file_to_class[file_path] = public_classes
                        for class_name in public_classes:
                            class_to_file[class_name] = file_path

        return file_to_class, class_to_file

    def get_identifier(self, node, types):
        """
        Retrieves the identifier node from the given node's children.

        Args:
            node (tree_sitter.Node): The node to search within.
            types (list): A list of node types to search for.

        Returns:
            tree_sitter.Node: The identifier node if found, otherwise None.
        """
        if node is None:
            return None

        for child in node.children:
            if child.type in types:
                return child

        return None

    def find_used_classes(self, file_path):
        """
        Parses a Java file and extracts all class or interface references used in it.

        Args:
            file_path (str): The path to the Java file.

        Returns:
            set: A set of class or interface names referenced in the file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                source_code = file.read()
        except (FileNotFoundError, UnicodeDecodeError) as e:
            print(f"Error processing {file_path}: {e}")
            return set()

        tree = self.parser.parse(bytes(source_code, "utf8"))
        class_references = set()

        def traverse(node):
            if node is None:
                return

            if node.type in ['field_declaration', 'formal_parameter', 'local_variable_declaration', 'call_expression']:
                identifier_node = self.get_identifier(node, ['type_identifier', 'generic_type', 'integral_type', 'scoped_type_identifier'])
                if identifier_node:
                    class_references.add(identifier_node.text.decode('utf-8'))
            else:
                for child in node.children:
                    traverse(child)

        traverse(tree.root_node)

        return class_references

    def get_dependency_graph(self, project_dir):
        """
        Constructs a dependency graph of Java files based on their class or interface usage.

        Args:
            project_dir (str): The root directory of the Java project.

        Returns:
            networkx.DiGraph: A directed graph where nodes are file paths and edges represent dependencies.
        """
        file_to_class, class_to_file = self.find_public_classes_in_project(project_dir)
        G = nx.DiGraph()

        # Add nodes for each Java file
        G.add_nodes_from(file_to_class.keys())

        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    used_classes = self.find_used_classes(file_path)
                    for class_name in used_classes:
                        if class_name in class_to_file and file_path != class_to_file[class_name]:
                            G.add_edge(file_path, class_to_file[class_name])

        return G

    def get_execution_order(self, project_dir):
        """
        Determines the execution order of Java files in a project based on their dependencies.

        Args:
            project_dir (str): The root directory of the Java project.

        Returns:
            tuple: 
                - networkx.DiGraph: The dependency graph of the project.
                - list: A list of lists, where each inner list contains files that can be executed in parallel.
        """
        G = self.get_dependency_graph(project_dir)
        sorted_tasks = list(nx.topological_sort(G))[::-1]

        def group_tasks_optimized(tasks, graph):
            """
            Groups tasks based on their dependencies in a directed acyclic graph (DAG).

            This function efficiently groups tasks into independent sets, ensuring that
            tasks within the same group do not have dependencies on each other. This is
            useful for determining parallel execution opportunities and visualizing task
            dependencies.

            Args:
                sorted_tasks: A list of tasks in topological order. Tasks should be listed
                    before their dependencies.
                G: A NetworkX DiGraph representing the task dependencies. An edge (u, v)
                    indicates that task u must be completed before task v.

            Returns:
                A list of lists, where each inner list represents a group of independent
                tasks.

            Raises:
                ValueError: If the input graph G is not a directed acyclic graph (DAG).

            Complexity:
                Time Complexity: O(n * m), where n is the number of tasks and m is the
                    average number of dependencies per task.
                Space Complexity: O(n) to store the task-to-group mapping.

            Example:
                >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 4), (3, 4)])
                >>> sorted_tasks = [1, 2, 3, 4]
                >>> group_tasks_optimized(sorted_tasks, G)
                [[1], [2, 3], [4]]  # Tasks 2 and 3 can be executed in parallel
            """
              
            grouped_tasks = []
            task_to_group = {}

            for task in tasks:
                group_number = -1

                # Determine the appropriate group for the task based on its predecessors
                for predecessor in graph.successors(task):
                    if predecessor in task_to_group:
                        group_number = max(group_number, task_to_group[predecessor])

                if group_number + 1 < len(grouped_tasks):
                    grouped_tasks[group_number + 1].append(task)
                else:
                    grouped_tasks.append([task])

                task_to_group[task] = group_number + 1

            return grouped_tasks

        grouped_tasks = group_tasks_optimized(sorted_tasks, G)

        return G, grouped_tasks