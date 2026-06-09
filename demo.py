from app.tools.resume_parser import parse_resume
from app.tools.jd_parser import parse_jd
from app.tools.match_score import calculate_match_score, generate_suggestions


def main():
    resume_path = "examples/resume.txt"
    jd_path = "examples/jd.txt"

    resume_result = parse_resume(resume_path)
    jd_result = parse_jd(jd_path)

    match_result = calculate_match_score(
        resume_skills=resume_result["skills"],
        jd_skills=jd_result["required_skills"]
    )

    suggestions = generate_suggestions(match_result["missing_skills"])

    print("=" * 50)
    print("简历-JD 匹配分析结果")
    print("=" * 50)

    print(f"\n匹配度：{match_result['match_score']}%")

    print("\n简历已有技能：")
    print("、".join(match_result["resume_skills"]))

    print("\n岗位要求技能：")
    print("、".join(match_result["jd_skills"]))

    print("\n匹配技能：")
    print("、".join(match_result["matched_skills"]))

    print("\n缺失技能：")
    print("、".join(match_result["missing_skills"]))

    print("\n优化建议：")
    for idx, suggestion in enumerate(suggestions, start=1):
        print(f"{idx}. {suggestion}")


if __name__ == "__main__":
    main()