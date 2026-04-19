# scientific_quality/annotation/annotation_guide.py
"""
Annotation rubric and guidelines for human experts.
This ensures consistent quality judgments across annotators.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class DimensionRubric:
    """Rubric for a single quality dimension."""
    name: str
    description: str
    criteria: Dict[int, str]  # score -> description
    questions: List[str]  # Questions annotators should ask themselves


class QualityRubric:
    """
    Complete rubric for KO quality annotation.
    
    Based on academic standards for content quality assessment.
    """
    
    DIMENSIONS = {
        "structural": DimensionRubric(
            name="Structural Quality",
            description="How well-organized and complete is the KO?",
            criteria={
                1: "Severely incomplete or chaotic structure. Missing critical fields (title, description, or content).",
                2: "Poor structure. Major sections missing or very disorganized.",
                3: "Adequate structure. All required fields present but could be better organized.",
                4: "Good structure. Well-organized with clear sections and appropriate metadata.",
                5: "Excellent structure. Complete, well-organized, with clear hierarchy and navigation cues.",
            },
            questions=[
                "Are all required fields present (title, description, content)?",
                "Is the content organized into logical sections?",
                "Are there clear headings or step markers?",
                "Is the length appropriate (not too short/long)?",
            ]
        ),
        
        "semantic": DimensionRubric(
            name="Semantic Quality",
            description="How clear, coherent, and useful is the content?",
            criteria={
                1: "Incoherent or unintelligible. Major clarity issues prevent understanding.",
                2: "Poor clarity. Multiple confusing passages or inconsistent information.",
                3: "Adequate clarity. Understandable but could be clearer or more focused.",
                4: "Good clarity. Clear, coherent, and generally useful information.",
                5: "Excellent clarity. Very well-written, actionable, and immediately useful.",
            },
            questions=[
                "Is the language clear and unambiguous?",
                "Are sentences of appropriate length?",
                "Does it provide actionable guidance (not just theory)?",
                "Is the information internally consistent?",
            ]
        ),
        
        "domain": DimensionRubric(
            name="Domain Quality",
            description="How accurate and relevant is the agricultural content?",
            criteria={
                1: "Not agricultural or factually incorrect.",
                2: "Weak agricultural relevance or significant inaccuracies.",
                3: "Moderately agricultural. Some relevance but not focused.",
                4: "Good agricultural content. Accurate and relevant.",
                5: "Excellent agricultural content. Highly accurate, relevant, and specialized.",
            },
            questions=[
                "Is this actually about agriculture?",
                "Are the facts accurate (to your knowledge)?",
                "Is it relevant to farmers/agricultural practitioners?",
                "Does it use appropriate agricultural terminology?",
            ]
        ),
        
        "functional": DimensionRubric(
            name="Functional Quality",
            description="How searchable and discoverable is the KO?",
            criteria={
                1: "Not discoverable. Poor keywords, no metadata overlap with content.",
                2: "Weak discoverability. Keywords present but poorly aligned.",
                3: "Adequate discoverability. Basic metadata present, moderate alignment.",
                4: "Good discoverability. Well-aligned metadata and content, good keywords.",
                5: "Excellent discoverability. Perfect alignment, comprehensive keywords, search-optimized.",
            },
            questions=[
                "Do keywords accurately reflect content?",
                "Does the title represent the content?",
                "Would this appear in relevant searches?",
                "Is there sufficient metadata for discovery?",
            ]
        ),
        
        "overall": DimensionRubric(
            name="Overall Quality",
            description="Your holistic assessment of this KO's value.",
            criteria={
                1: "Not worth keeping. Severe issues in multiple dimensions.",
                2: "Poor quality. Major issues, limited value.",
                3: "Average quality. Acceptable but not exceptional.",
                4: "Good quality. Worth recommending to farmers.",
                5: "Excellent quality. Highly valuable, would strongly recommend.",
            },
            questions=[
                "Would you recommend this KO to a farmer?",
                "Does it provide real value?",
                "Is it trustworthy?",
            ]
        ),
    }
    
    @classmethod
    def get_rubric(cls, dimension: str) -> DimensionRubric:
        """Get rubric for a specific dimension."""
        if dimension not in cls.DIMENSIONS:
            raise ValueError(f"Unknown dimension: {dimension}. Choose from {list(cls.DIMENSIONS.keys())}")
        return cls.DIMENSIONS[dimension]
    
    @classmethod
    def get_all_dimensions(cls) -> List[str]:
        """Get list of all quality dimensions."""
        return list(cls.DIMENSIONS.keys())


class AnnotationGuidelines:
    """
    Complete annotation guidelines for expert annotators.
    """
    
    GENERAL_GUIDELINES = """
    # General Annotation Guidelines

    ## Before You Start
    1. Read the complete rubric for each dimension
    2. Review 5-10 example KOs with consensus scores
    3. When in doubt, use your expert judgment

    ## During Annotation
    1. Read the entire KO before scoring
    2. Score each dimension independently
    3. Use the full 1-5 scale (don't cluster in middle)
    4. Take breaks every 20 KOs to maintain consistency
    5. Flag any KO that seems ambiguous

    ## Common Pitfalls
    - **Recency bias**: Don't let the last KO influence the next
    - **Halo effect**: Score each dimension independently
    - **Central tendency**: Use the extremes when warranted
    - **Fatigue**: Take breaks, annotate in batches

    ## Flagging Issues
    If you encounter:
    - Non-English content (mark language)
    - Broken/missing content (note in comments)
    - Unclear scoring (discuss with other annotators)
    """
    
    @classmethod
    def export_markdown(cls, output_path: str):
        """Export complete guidelines as markdown for annotators."""
        with open(output_path, 'w') as f:
            f.write(cls.GENERAL_GUIDELINES)
            f.write("\n\n## Dimension-Specific Rubrics\n\n")
            
            for dim_name, rubric in QualityRubric.DIMENSIONS.items():
                f.write(f"\n### {rubric.name}\n\n")
                f.write(f"{rubric.description}\n\n")
                
                f.write("**Scoring Criteria:**\n\n")
                for score, desc in rubric.criteria.items():
                    f.write(f"- **{score}**: {desc}\n")
                
                f.write("\n**Questions to Ask:**\n\n")
                for q in rubric.questions:
                    f.write(f"- {q}\n")
                
                f.write("\n---\n")
