from setuptools import setup, find_packages

setup(
    name="wild",
    version="1.3.0",
    packages=find_packages(),
    package_data={
        "diffgraph": [
            "schema/diffgraph-v2.schema.json",
            "schema/diffgraph-v2.structural.example.json",
        ]
    },
    install_requires=[
        "click>=8.1.7",
        "click-spinner>=0.1.10",
        "jsonschema>=4.18",
        'networkx>=3.4.2,<3.5; python_version < "3.11"',
        'networkx>=3.5; python_version >= "3.11"',
        "openai-agents>=0.0.17",
        "python-dotenv>=1.0.0",
        "tree-sitter-language-pack>=0.10.0,<2",
        "tree-sitter>=0.23,<1",
    ],
    entry_points={
        "console_scripts": [
            "wild=diffgraph.cli:main",
        ],
    },
    python_requires=">=3.10",
    author="WildestAI Team",
    description="AI-powered git-wrapper CLI tool with visualizing diffs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
