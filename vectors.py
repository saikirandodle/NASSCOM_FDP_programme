import math


def l2_norm(vector):
	"""Return the Euclidean (L2) norm of a vector."""
	return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(vector_a, vector_b):
	"""Return cosine similarity between two vectors."""
	dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
	norm_a = l2_norm(vector_a)
	norm_b = l2_norm(vector_b)
	if norm_a == 0 or norm_b == 0:
		return 0.0
	return dot_product / (norm_a * norm_b)


def similarity_label(score):
	"""Provide a simple interpretation for cosine similarity."""
	if score >= 0.98:
		return "very similar performance pattern"
	if score >= 0.90:
		return "similar performance pattern"
	if score >= 0.75:
		return "moderately similar pattern"
	return "different performance pattern"


# Student exam scores in [math, science, english]
student_a = [92, 88, 84]
student_b = [78, 82, 80]
student_c = [95, 70, 60]

students = {
	"Student A": student_a,
	"Student B": student_b,
	"Student C": student_c,
}

print("Student score vectors [math, science, english]:")
for name, scores in students.items():
	print(f"{name}: {scores}")

print("\nL2 norms (overall performance magnitude):")
for name, scores in students.items():
	print(f"{name}: {l2_norm(scores):.2f}")

pair_scores = {
	"A-B": cosine_similarity(student_a, student_b),
	"A-C": cosine_similarity(student_a, student_c),
	"B-C": cosine_similarity(student_b, student_c),
}

print("\nCosine similarity between pairs:")
for pair, score in pair_scores.items():
	print(f"{pair}: {score:.4f} ({similarity_label(score)})")

print("\nInterpretation:")
print("- Larger L2 norm means stronger overall scores across the three subjects.")
print("- Student A has the strongest overall performance.")
print("- Student B is close to Student A in score pattern (balanced across subjects).")
print("- Student C differs more from B due to stronger math but lower english/science.")
