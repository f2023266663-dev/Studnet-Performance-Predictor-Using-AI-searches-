"""
EduPredict AI - Classical AI Student Performance Prediction & Intervention System
Classical AI: Rule-Based Expert System + Forward Chaining + A* Search
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import heapq
import random
import copy

# ══════════════════════════════════════════════════════════════════════════════
# 1. STUDENT PROFILE DATACLASS (32 Features)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StudentProfile:
    # School & Demographics
    school: str = 'GP'       # 'GP' or 'MS'
    sex: str = 'M'           # 'F' or 'M'
    age: int = 17            # 15–22
    address: str = 'U'       # 'U' (Urban) or 'R' (Rural)
    famsize: str = 'GT3'     # 'LE3' or 'GT3'

    # Parental & Family Information
    Pstatus: str = 'T'       # 'T' (Together) or 'A' (Apart)
    Medu: int = 2            # 0–4
    Fedu: int = 2            # 0–4
    famsup: str = 'yes'      # 'yes' or 'no'
    paid: str = 'no'         # 'yes' or 'no'

    # School Life
    activities: str = 'no'  # 'yes' or 'no'
    nursery: str = 'yes'     # 'yes' or 'no'
    higher: str = 'yes'      # 'yes' or 'no'
    internet: str = 'yes'    # 'yes' or 'no'
    romantic: str = 'no'     # 'yes' or 'no'

    # Academic & Study Habits
    studytime: int = 2       # 1–4
    failures: int = 0        # 0–4+
    absences: int = 4        # 0–93

    # Social & Behavioral
    goout: int = 3           # 1–5
    Dalc: int = 1            # 1–5
    Walc: int = 1            # 1–5
    health: int = 3          # 1–5
    traveltime: int = 1      # 1–4

    # Course-Specific
    reason: str = 'course'   # 'home','reputation','course','other'
    guardian: str = 'mother' # 'mother','father','other'

    # Family Relations
    famrel: int = 4          # 1–5
    freetime: int = 3        # 1–5
    schoolsup: str = 'no'    # 'yes' or 'no'

    # Grades (Target = G3)
    G1: int = 0
    G2: int = 0
    G3: int = 0

    def get_failed_tests(self) -> List[Tuple[str, int]]:
        failed = []
        if self.G1 < 10: failed.append(('G1 (First Period)', self.G1))
        if self.G2 < 10: failed.append(('G2 (Second Period)', self.G2))
        if self.G3 < 10: failed.append(('G3 (Final Grade)', self.G3))
        return failed

    def has_triple_failure(self) -> bool:
        # Only flag triple failure when grades were actually recorded (not zero/blank)
        grades_recorded = (self.G1 > 0 or self.G2 > 0 or self.G3 > 0)
        return grades_recorded and self.G1 < 10 and self.G2 < 10 and self.G3 < 10


# ══════════════════════════════════════════════════════════════════════════════
# 2. RULE-BASED EXPERT SYSTEM (25-30 IF-THEN Rules)
# ══════════════════════════════════════════════════════════════════════════════

class RuleBase:
    """
    25-30 IF-THEN rules covering all risk categories.
    Each rule: condition function → (conclusion, weight, category, explanation)
    """

    RULES = [
        # ── ACADEMIC RISK RULES ──────────────────────────────────────────────
        {
            'id': 'R01',
            'category': 'critical',
            'name': 'Triple Failure',
            'condition': lambda s: (s.G1 > 0 or s.G2 > 0 or s.G3 > 0) and s.G1 < 10 and s.G2 < 10 and s.G3 < 10,
            'conclusion': 'CRITICAL_FAIL',
            'weight': 1.0,
            'explanation': 'Failed ALL THREE tests. MUST REPEAT this course.'
        },
        {
            'id': 'R02',
            'category': 'critical',
            'name': 'High Past Failures',
            'condition': lambda s: s.failures >= 3,
            'conclusion': 'HIGH_RISK',
            'weight': 0.90,
            'explanation': 'Multiple past failures indicate significant academic struggle.'
        },
        {
            'id': 'R03',
            'category': 'critical',
            'name': 'Very Low Study Time',
            'condition': lambda s: s.studytime == 1 and s.failures >= 2,
            'conclusion': 'HIGH_RISK',
            'weight': 0.85,
            'explanation': 'Minimal study time combined with past failures creates a critical risk.'
        },
        {
            'id': 'R04',
            'category': 'critical',
            'name': 'High Absence Rate',
            'condition': lambda s: s.absences >= 20,
            'conclusion': 'HIGH_RISK',
            'weight': 0.80,
            'explanation': 'High absence rate indicates severe disengagement from school.'
        },
        {
            'id': 'R05',
            'category': 'critical',
            'name': 'Failed Final Grade',
            'condition': lambda s: s.G3 < 10 and s.G3 > 0,
            'conclusion': 'FAIL',
            'weight': 0.95,
            'explanation': 'Final grade G3 is below passing threshold of 10.'
        },
        {
            'id': 'R06',
            'category': 'moderate',
            'name': 'Moderate Past Failures',
            'condition': lambda s: s.failures == 2,
            'conclusion': 'MODERATE_RISK',
            'weight': 0.65,
            'explanation': 'Two past failures suggest a recurring pattern of academic difficulty.'
        },
        {
            'id': 'R07',
            'category': 'moderate',
            'name': 'Low Study Time',
            'condition': lambda s: s.studytime == 1,
            'conclusion': 'MODERATE_RISK',
            'weight': 0.55,
            'explanation': 'Less than 2 hours of study per week is insufficient for most subjects.'
        },
        {
            'id': 'R08',
            'category': 'moderate',
            'name': 'Moderate Absences',
            'condition': lambda s: 10 <= s.absences < 20,
            'conclusion': 'MODERATE_RISK',
            'weight': 0.50,
            'explanation': 'Moderate absenteeism is starting to affect academic engagement.'
        },
        {
            'id': 'R09',
            'category': 'moderate',
            'name': 'Single Past Failure',
            'condition': lambda s: s.failures == 1,
            'conclusion': 'LOW_RISK',
            'weight': 0.35,
            'explanation': 'One past failure — manageable but worth monitoring.'
        },
        # ── SOCIAL RISK RULES ────────────────────────────────────────────────
        {
            'id': 'R10',
            'category': 'critical',
            'name': 'High Social + Alcohol',
            'condition': lambda s: s.goout >= 4 and s.Dalc >= 3,
            'conclusion': 'HIGH_RISK',
            'weight': 0.75,
            'explanation': 'High social activity and weekday alcohol consumption harm academic focus.'
        },
        {
            'id': 'R11',
            'category': 'moderate',
            'name': 'High Weekend Alcohol',
            'condition': lambda s: s.Walc >= 4,
            'conclusion': 'MODERATE_RISK',
            'weight': 0.50,
            'explanation': 'High weekend alcohol use often disrupts study patterns on Mondays.'
        },
        {
            'id': 'R12',
            'category': 'moderate',
            'name': 'Very High Social Activity',
            'condition': lambda s: s.goout == 5,
            'conclusion': 'MODERATE_RISK',
            'weight': 0.45,
            'explanation': 'Maximum social activity leaves minimal time for academic work.'
        },
        {
            'id': 'R13',
            'category': 'moderate',
            'name': 'High Weekday Alcohol',
            'condition': lambda s: s.Dalc >= 4,
            'conclusion': 'MODERATE_RISK',
            'weight': 0.60,
            'explanation': 'High weekday alcohol consumption directly impairs learning capacity.'
        },
        {
            'id': 'R14',
            'category': 'moderate',
            'name': 'Romantic Relationship + Low Grades',
            'condition': lambda s: s.romantic == 'yes' and s.failures >= 1 and s.studytime <= 2,
            'conclusion': 'MODERATE_RISK',
            'weight': 0.35,
            'explanation': 'Romantic involvement combined with poor study habits increases distraction risk.'
        },
        # ── FAMILY RISK RULES ────────────────────────────────────────────────
        {
            'id': 'R15',
            'category': 'moderate',
            'name': 'No Family Support + Failing',
            'condition': lambda s: s.famsup == 'no' and s.failures >= 1,
            'conclusion': 'MODERATE_RISK',
            'weight': 0.50,
            'explanation': 'Lack of family educational support with prior failures increases dropout risk.'
        },
        {
            'id': 'R16',
            'category': 'moderate',
            'name': 'Parents Apart + Low Performance',
            'condition': lambda s: s.Pstatus == 'A' and s.failures >= 2,
            'conclusion': 'MODERATE_RISK',
            'weight': 0.40,
            'explanation': 'Separated parents combined with academic failure may indicate home instability.'
        },
        {
            'id': 'R17',
            'category': 'moderate',
            'name': 'Low Parental Education',
            'condition': lambda s: s.Medu <= 1 and s.Fedu <= 1,
            'conclusion': 'MODERATE_RISK',
            'weight': 0.35,
            'explanation': 'Low parental education limits the academic support available at home.'
        },
        {
            'id': 'R18',
            'category': 'moderate',
            'name': 'Poor Family Relations',
            'condition': lambda s: s.famrel <= 2,
            'conclusion': 'MODERATE_RISK',
            'weight': 0.40,
            'explanation': 'Poor family relationships create a stressful environment that hinders learning.'
        },
        # ── ASPIRATION & MOTIVATION RULES ────────────────────────────────────
        {
            'id': 'R19',
            'category': 'positive',
            'name': 'Higher Education Goal',
            'condition': lambda s: s.higher == 'yes',
            'conclusion': 'POSITIVE',
            'weight': -0.25,
            'explanation': 'Student aspires to higher education — a strong motivational protective factor.'
        },
        {
            'id': 'R20',
            'category': 'positive',
            'name': 'No Aspiration Risk',
            'condition': lambda s: s.higher == 'no' and s.studytime <= 2,
            'conclusion': 'LOW_RISK',
            'weight': 0.30,
            'explanation': 'No higher education goal combined with low study time signals disengagement.'
        },
        # ── HEALTH & WELLBEING RULES ─────────────────────────────────────────
        {
            'id': 'R21',
            'category': 'moderate',
            'name': 'Poor Health',
            'condition': lambda s: s.health <= 2,
            'conclusion': 'MODERATE_RISK',
            'weight': 0.40,
            'explanation': 'Poor self-reported health may increase absenteeism and reduce concentration.'
        },
        {
            'id': 'R22',
            'category': 'positive',
            'name': 'Good Health',
            'condition': lambda s: s.health >= 4,
            'conclusion': 'POSITIVE',
            'weight': -0.10,
            'explanation': 'Good health supports consistent school attendance and cognitive performance.'
        },
        {
            'id': 'R23',
            'category': 'moderate',
            'name': 'Long Travel Time',
            'condition': lambda s: s.traveltime >= 3,
            'conclusion': 'MODERATE_RISK',
            'weight': 0.30,
            'explanation': 'Long commute reduces energy and time available for studying after school.'
        },
        # ── PROTECTIVE / POSITIVE RULES ──────────────────────────────────────
        {
            'id': 'R24',
            'category': 'positive',
            'name': 'Strong Academic Foundation',
            'condition': lambda s: s.studytime >= 3 and s.failures == 0,
            'conclusion': 'POSITIVE',
            'weight': -0.40,
            'explanation': 'High study time with zero failures demonstrates strong academic discipline.'
        },
        {
            'id': 'R25',
            'category': 'positive',
            'name': 'Family Support + Good Study',
            'condition': lambda s: s.famsup == 'yes' and s.studytime >= 2,
            'conclusion': 'POSITIVE',
            'weight': -0.20,
            'explanation': 'Family educational support combined with adequate study time is a strong protective factor.'
        },
        {
            'id': 'R26',
            'category': 'positive',
            'name': 'Internet Access + Study',
            'condition': lambda s: s.internet == 'yes' and s.studytime >= 2,
            'conclusion': 'POSITIVE',
            'weight': -0.10,
            'explanation': 'Internet access enables self-directed learning and access to educational resources.'
        },
        {
            'id': 'R27',
            'category': 'positive',
            'name': 'Paid Tutoring',
            'condition': lambda s: s.paid == 'yes',
            'conclusion': 'POSITIVE',
            'weight': -0.20,
            'explanation': 'Paid extra classes provide targeted academic support and structured revision.'
        },
        {
            'id': 'R28',
            'category': 'positive',
            'name': 'High Parental Education',
            'condition': lambda s: s.Medu >= 3 or s.Fedu >= 3,
            'conclusion': 'POSITIVE',
            'weight': -0.15,
            'explanation': 'Higher parental education correlates with stronger home academic support.'
        },
        {
            'id': 'R29',
            'category': 'moderate',
            'name': 'School Support',
            'condition': lambda s: s.schoolsup == 'yes',
            'conclusion': 'POSITIVE',
            'weight': -0.15,
            'explanation': 'School-provided educational support helps students at risk of falling behind.'
        },
        {
            'id': 'R30',
            'category': 'positive',
            'name': 'Zero Absence Record',
            'condition': lambda s: s.absences == 0,
            'conclusion': 'POSITIVE',
            'weight': -0.20,
            'explanation': 'Perfect attendance record demonstrates full commitment to education.'
        },
    ]

    @classmethod
    def _grade_anchored_probability(cls, student: StudentProfile, behavior_risk: float) -> Tuple[float, str, str]:
        """
        Grade-Anchored Confidence System.

        Logic:
        - If G3 is available (> 0), it is ground truth.  The grade score provides
          a strong anchor for the probability; behavioral risk adjusts it modestly.
        - If G3 = 0 (not yet available / predictive mode), the probability is
          derived purely from behavioral risk rules.

        Returns: (probability, confidence_tier, confidence_note)
        """
        GRADE_MAX = 20.0
        PASS_THRESHOLD = 10

        # ── CASE 1: Grade is available ──────────────────────────────────────
        if student.G3 > 0:
            # Grade contributes 80% of the signal; behavior risk adjusts the other 20%
            grade_score = student.G3 / GRADE_MAX          # 0.0 – 1.0

            # Behavior adjustment: net risk (can be negative if many positive rules)
            # Clamped to ±0.2 so it nudges but never overrides the grade
            behavior_adj = max(-0.20, min(0.20, -behavior_risk * 0.20))

            raw_prob = (grade_score * 0.80) + (0.50 * 0.20) + behavior_adj
            probability = max(5.0, min(98.0, raw_prob * 100))

            # Confidence tier based on grade margin from threshold
            margin = student.G3 - PASS_THRESHOLD          # negative = failing
            if student.G3 >= PASS_THRESHOLD:
                if margin >= 6:
                    tier = "HIGH CONFIDENCE PASS"
                    note = f"Grade {student.G3}/20 is {margin} points above the passing threshold."
                elif margin >= 3:
                    tier = "MODERATE CONFIDENCE PASS"
                    note = f"Grade {student.G3}/20 passes comfortably. Behavioral risks noted but grade is secure."
                else:
                    tier = "BORDERLINE PASS"
                    note = f"Grade {student.G3}/20 just clears the threshold. Behavioral risks could be a concern next term."
            else:
                if margin <= -6:
                    tier = "HIGH CONFIDENCE FAIL"
                    note = f"Grade {student.G3}/20 is {abs(margin)} points below passing. Significant intervention needed."
                elif margin <= -3:
                    tier = "MODERATE CONFIDENCE FAIL"
                    note = f"Grade {student.G3}/20 is below passing. Targeted support recommended."
                else:
                    tier = "BORDERLINE FAIL"
                    note = f"Grade {student.G3}/20 narrowly misses the threshold. Small improvements could make the difference."

        # ── CASE 2: Predictive mode — no grade yet ──────────────────────────
        else:
            base_risk = max(0.0, min(1.0, behavior_risk))
            raw_prob = 1.0 - base_risk
            probability = max(5.0, min(95.0, raw_prob * 100))

            if probability >= 70:
                tier = "LIKELY PASS (Predicted)"
                note = "No grade available yet. Prediction based on behavioral and family factors."
            elif probability >= 40:
                tier = "AT RISK (Predicted)"
                note = "No grade available yet. Moderate risk profile — early intervention recommended."
            else:
                tier = "HIGH RISK (Predicted)"
                note = "No grade available yet. High behavioral risk — immediate support advised."

        return round(probability, 1), tier, note

    @classmethod
    def evaluate_rules(cls, student: StudentProfile) -> Dict:
        fired_rules = []
        total_risk_weight = 0.0
        critical = []
        moderate = []
        positive = []

        for rule in cls.RULES:
            try:
                if rule['condition'](student):
                    fired_rules.append(rule)
                    total_risk_weight += rule['weight']
                    if rule['category'] == 'critical':
                        critical.append(rule['explanation'])
                    elif rule['category'] == 'moderate':
                        moderate.append(rule['explanation'])
                    elif rule['category'] == 'positive':
                        positive.append(rule['explanation'])
            except Exception:
                continue

        # ── Grade-Anchored Probability (replaces raw risk formula) ──────────
        pass_probability, confidence_tier, confidence_note = cls._grade_anchored_probability(
            student, total_risk_weight
        )

        # ── Determine prediction ─────────────────────────────────────────────
        if student.G3 >= 10:
            prediction = 'PASS'
        elif student.has_triple_failure():
            prediction = 'CRITICAL_FAIL'
        elif student.G3 > 0:
            prediction = 'FAIL'
        elif total_risk_weight >= 0.3:
            prediction = 'FAIL'
        else:
            prediction = 'PASS'

        return {
            'prediction': prediction,
            'pass_probability': pass_probability,
            'confidence_tier': confidence_tier,
            'confidence_note': confidence_note,
            'rules_applied': len(fired_rules),
            'total_risk_weight': total_risk_weight,
            'critical_factors': critical,
            'moderate_factors': moderate,
            'positive_factors': positive,
            'fired_rules': fired_rules,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. FORWARD CHAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ForwardChainingEngine:
    """
    Infers higher-level facts from basic student attributes.
    Represents classical AI knowledge inference.
    """

    INFERENCE_RULES = [
        {
            'condition': lambda s: s.studytime >= 3 and s.failures == 0,
            'insight': 'Strong academic foundation — student demonstrates consistent discipline.'
        },
        {
            'condition': lambda s: s.famsup == 'yes' and s.Medu >= 3,
            'insight': 'Strong family support system — educated parent can provide direct academic help.'
        },
        {
            'condition': lambda s: s.higher == 'yes' and s.studytime >= 2,
            'insight': 'Goal-oriented student — aspiration for higher education drives consistent effort.'
        },
        {
            'condition': lambda s: s.absences >= 15 and s.failures >= 1,
            'insight': 'Attendance-failure cycle detected — absenteeism is compounding academic difficulty.'
        },
        {
            'condition': lambda s: s.Dalc >= 3 and s.goout >= 4,
            'insight': 'High-risk lifestyle pattern — social behavior is interfering with academic engagement.'
        },
        {
            'condition': lambda s: s.internet == 'yes' and s.studytime >= 2 and s.higher == 'yes',
            'insight': 'Digitally-enabled learner — good access to self-study resources.'
        },
        {
            'condition': lambda s: s.paid == 'yes' and s.failures >= 1,
            'insight': 'Intervention already in progress — student or family has recognised the academic gap.'
        },
        {
            'condition': lambda s: s.health <= 2 and s.absences >= 10,
            'insight': 'Health-attendance link — poor health appears to be driving school absences.'
        },
        {
            'condition': lambda s: s.Pstatus == 'A' and s.famsup == 'no',
            'insight': 'Minimal home support — separated parents with no educational support is a vulnerability.'
        },
        {
            'condition': lambda s: s.traveltime >= 3 and s.studytime <= 2,
            'insight': 'Long commute limiting study time — transport burden is reducing available study hours.'
        },
        {
            'condition': lambda s: s.famrel >= 4 and s.famsup == 'yes',
            'insight': 'Stable and supportive family environment — a key protective factor for academic success.'
        },
        {
            'condition': lambda s: s.romantic == 'yes' and s.goout >= 4,
            'insight': 'Social commitments are high — time management may be a critical issue.'
        },
        {
            'condition': lambda s: s.schoolsup == 'yes' and s.failures >= 1,
            'insight': 'School support activated — existing academic safety net may help prevent further failure.'
        },
        {
            'condition': lambda s: s.studytime == 1 and s.higher == 'no',
            'insight': 'Low motivation and effort — student shows neither study engagement nor academic ambition.'
        },
        {
            'condition': lambda s: s.absences == 0 and s.studytime >= 2,
            'insight': 'Exemplary attendance and study habits — maximum engagement with the educational environment.'
        },
    ]

    @classmethod
    def infer(cls, student: StudentProfile) -> List[str]:
        insights = []
        for rule in cls.INFERENCE_RULES:
            try:
                if rule['condition'](student):
                    insights.append(rule['insight'])
            except Exception:
                continue
        return insights


# ══════════════════════════════════════════════════════════════════════════════
# 4. A* SEARCH ALGORITHM — INTERVENTION FINDER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchNode:
    student: StudentProfile
    actions: List[str] = field(default_factory=list)
    cost: int = 0
    heuristic: float = 0.0

    @property
    def f_score(self):
        return self.cost + self.heuristic

    def __lt__(self, other):
        return self.f_score < other.f_score


class InterventionSearcher:
    """
    A* Search to find the optimal (minimum-step) path from FAIL to PASS.
    f(n) = g(n) [cost = steps taken] + h(n) [heuristic = risk estimate]
    """

    ACTIONS = [
        {
            'name': 'increase_studytime',
            'label': lambda s: f"Increase studytime from {s.studytime} to {min(s.studytime+1,4)}",
            'apply': lambda s: setattr(s, 'studytime', min(s.studytime + 1, 4)),
            'applicable': lambda s: s.studytime < 4,
        },
        {
            'name': 'reduce_absences',
            'label': lambda s: f"Reduce absences from {s.absences} to {max(s.absences-5,0)}",
            'apply': lambda s: setattr(s, 'absences', max(s.absences - 5, 0)),
            'applicable': lambda s: s.absences >= 5,
        },
        {
            'name': 'reduce_goout',
            'label': lambda s: f"Reduce social activities (goout) from {s.goout} to {max(s.goout-1,1)}",
            'apply': lambda s: setattr(s, 'goout', max(s.goout - 1, 1)),
            'applicable': lambda s: s.goout > 1,
        },
        {
            'name': 'reduce_dalc',
            'label': lambda s: f"Reduce weekday alcohol (Dalc) from {s.Dalc} to {max(s.Dalc-1,1)}",
            'apply': lambda s: setattr(s, 'Dalc', max(s.Dalc - 1, 1)),
            'applicable': lambda s: s.Dalc > 1,
        },
        {
            'name': 'reduce_walc',
            'label': lambda s: f"Reduce weekend alcohol (Walc) from {s.Walc} to {max(s.Walc-1,1)}",
            'apply': lambda s: setattr(s, 'Walc', max(s.Walc - 1, 1)),
            'applicable': lambda s: s.Walc > 1,
        },
        {
            'name': 'add_famsup',
            'label': lambda s: "Engage family educational support",
            'apply': lambda s: setattr(s, 'famsup', 'yes'),
            'applicable': lambda s: s.famsup == 'no',
        },
        {
            'name': 'add_paid',
            'label': lambda s: "Enrol in paid tutoring classes",
            'apply': lambda s: setattr(s, 'paid', 'yes'),
            'applicable': lambda s: s.paid == 'no',
        },
        {
            'name': 'add_schoolsup',
            'label': lambda s: "Enrol in school educational support programme",
            'apply': lambda s: setattr(s, 'schoolsup', 'yes'),
            'applicable': lambda s: s.schoolsup == 'no',
        },
        {
            'name': 'set_higher',
            'label': lambda s: "Set academic goal: pursue higher education",
            'apply': lambda s: setattr(s, 'higher', 'yes'),
            'applicable': lambda s: s.higher == 'no',
        },
    ]

    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth

    def calculate_heuristic(self, student: StudentProfile) -> float:
        risk_count = 0
        if student.studytime <= 1: risk_count += 3
        elif student.studytime == 2: risk_count += 1
        if student.absences >= 20: risk_count += 3
        elif student.absences >= 10: risk_count += 2
        elif student.absences >= 5: risk_count += 1
        if student.failures >= 3: risk_count += 3
        elif student.failures >= 2: risk_count += 2
        elif student.failures >= 1: risk_count += 1
        if student.Dalc >= 4: risk_count += 2
        elif student.Dalc >= 3: risk_count += 1
        if student.goout >= 4: risk_count += 1
        if student.famsup == 'no': risk_count += 1
        if student.higher == 'no': risk_count += 1
        return risk_count / 10.0

    def is_passing(self, student: StudentProfile) -> bool:
        result = RuleBase.evaluate_rules(student)
        return result['prediction'] in ('PASS',) and result['total_risk_weight'] < 0.3

    def get_successors(self, node: SearchNode) -> List[SearchNode]:
        successors = []
        for action in self.ACTIONS:
            try:
                if action['applicable'](node.student):
                    new_student = copy.deepcopy(node.student)
                    label = action['label'](new_student)
                    action['apply'](new_student)
                    new_heuristic = self.calculate_heuristic(new_student)
                    new_node = SearchNode(
                        student=new_student,
                        actions=node.actions + [label],
                        cost=node.cost + 1,
                        heuristic=new_heuristic
                    )
                    successors.append(new_node)
            except Exception:
                continue
        return successors

    def search(self, student: StudentProfile) -> Tuple[Optional[SearchNode], int]:
        if student.has_triple_failure():
            return None, 0

        initial_h = self.calculate_heuristic(student)
        start = SearchNode(
            student=copy.deepcopy(student),
            actions=[],
            cost=0,
            heuristic=initial_h
        )

        open_heap = []
        heapq.heappush(open_heap, (start.f_score, id(start), start))
        visited = set()
        nodes_explored = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            nodes_explored += 1

            state_key = (
                current.student.studytime,
                current.student.absences,
                current.student.goout,
                current.student.Dalc,
                current.student.Walc,
                current.student.famsup,
                current.student.paid,
                current.student.schoolsup,
                current.student.higher,
            )

            if state_key in visited:
                continue
            visited.add(state_key)

            if self.is_passing(current.student):
                return current, nodes_explored

            if current.cost >= self.max_depth:
                continue

            for successor in self.get_successors(current):
                heapq.heappush(open_heap, (successor.f_score, id(successor), successor))

        return None, nodes_explored


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN EduPredictAI SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class EduPredictAI:

    def __init__(self):
        self.searcher = InterventionSearcher(max_depth=8)

    def predict(self, student: StudentProfile) -> Dict:
        rule_results = RuleBase.evaluate_rules(student)
        insights = ForwardChainingEngine.infer(student)
        rule_results['insights'] = insights
        rule_results['triple_failure'] = student.has_triple_failure()
        rule_results['failed_tests'] = student.get_failed_tests()
        return rule_results

    def find_intervention(self, student: StudentProfile) -> Dict:
        result = self.predict(student)

        if result['prediction'] == 'PASS':
            return {
                'status': 'PASS',
                'message': '✅ Student is already predicted to PASS. No intervention needed.',
                'steps': [],
                'nodes_explored': 0,
                'final_prediction': 'PASS'
            }

        if student.has_triple_failure():
            return {
                'status': 'CRITICAL_FAIL',
                'message': '⚠️⚠️⚠️ CRITICAL: YOU MUST REPEAT THIS COURSE',
                'steps': [
                    'Score ≥ 10/20 on ALL tests in the repeated course',
                    f'Increase studytime to at least 3/4 (currently {student.studytime}/4)',
                    f'Reduce absences to under 10 days (currently {student.absences} days)',
                    'Enrol in tutoring or extra academic help sessions',
                    f'Reduce social activities (currently {student.goout}/5)',
                    f'Reduce alcohol consumption (currently Dalc={student.Dalc}/5)',
                ],
                'nodes_explored': 0,
                'final_prediction': 'PASS (After Repeat Attempt)'
            }

        solution, nodes_explored = self.searcher.search(student)

        if solution:
            return {
                'status': 'INTERVENTION_FOUND',
                'message': f'✅ Intervention found! {len(solution.actions)} change(s) needed.',
                'steps': solution.actions,
                'nodes_explored': nodes_explored,
                'final_prediction': 'PASS'
            }
        else:
            return {
                'status': 'CRITICAL',
                'message': '❌ Basic behavioral changes insufficient. Intensive counseling required.',
                'steps': [
                    'Engage school counselor for personalised support plan',
                    'Investigate underlying health, family, or motivational barriers',
                    'Consider reduced course load while building academic foundations',
                ],
                'nodes_explored': nodes_explored,
                'final_prediction': 'REQUIRES SPECIALIST INTERVENTION'
            }


# ══════════════════════════════════════════════════════════════════════════════
# 6. STUDENT GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def create_sample_student() -> StudentProfile:
    """Test Case 1: Passing student"""
    return StudentProfile(
        school='GP', sex='F', age=16, address='U', famsize='GT3',
        Pstatus='T', Medu=3, Fedu=2, famsup='yes', paid='no',
        activities='yes', nursery='yes', higher='yes', internet='yes', romantic='no',
        studytime=2, failures=1, absences=12,
        goout=2, Dalc=1, Walc=2, health=4, traveltime=1,
        reason='course', guardian='mother', famrel=4, freetime=3, schoolsup='no',
        G1=12, G2=11, G3=10
    )

def create_failing_student() -> StudentProfile:
    """Test Case 2: Failing student needing intervention"""
    return StudentProfile(
        school='MS', sex='M', age=17, address='U', famsize='GT3',
        Pstatus='T', Medu=2, Fedu=1, famsup='no', paid='no',
        activities='no', nursery='no', higher='no', internet='yes', romantic='yes',
        studytime=1, failures=2, absences=20,
        goout=4, Dalc=3, Walc=4, health=3, traveltime=2,
        reason='other', guardian='mother', famrel=3, freetime=4, schoolsup='no',
        G1=12, G2=8, G3=7
    )

def create_triple_failure_student() -> StudentProfile:
    """Test Case 3: Triple failure — must repeat course"""
    return StudentProfile(
        school='MS', sex='M', age=18, address='R', famsize='GT3',
        Pstatus='A', Medu=1, Fedu=1, famsup='no', paid='no',
        activities='no', nursery='no', higher='no', internet='no', romantic='yes',
        studytime=1, failures=3, absences=25,
        goout=5, Dalc=4, Walc=5, health=2, traveltime=3,
        reason='other', guardian='other', famrel=2, freetime=5, schoolsup='no',
        G1=5, G2=4, G3=3
    )

def generate_random_student() -> StudentProfile:
    """Generate a random student for demo purposes"""
    return StudentProfile(
        school=random.choice(['GP', 'MS']),
        sex=random.choice(['F', 'M']),
        age=random.randint(15, 20),
        address=random.choice(['U', 'R']),
        famsize=random.choice(['LE3', 'GT3']),
        Pstatus=random.choice(['T', 'A']),
        Medu=random.randint(0, 4),
        Fedu=random.randint(0, 4),
        famsup=random.choice(['yes', 'no']),
        paid=random.choice(['yes', 'no']),
        activities=random.choice(['yes', 'no']),
        nursery=random.choice(['yes', 'no']),
        higher=random.choice(['yes', 'no']),
        internet=random.choice(['yes', 'no']),
        romantic=random.choice(['yes', 'no']),
        studytime=random.randint(1, 4),
        failures=random.randint(0, 3),
        absences=random.randint(0, 30),
        goout=random.randint(1, 5),
        Dalc=random.randint(1, 5),
        Walc=random.randint(1, 5),
        health=random.randint(1, 5),
        traveltime=random.randint(1, 4),
        reason=random.choice(['home', 'reputation', 'course', 'other']),
        guardian=random.choice(['mother', 'father', 'other']),
        famrel=random.randint(1, 5),
        freetime=random.randint(1, 5),
        schoolsup=random.choice(['yes', 'no']),
        G1=random.randint(4, 18),
        G2=random.randint(4, 18),
        G3=random.randint(3, 18),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7. PRINT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def print_student_profile(student: StudentProfile):
    print("\n📋 Student Profile Summary:")
    print(f"  School: {student.school}, Age: {student.age}, Sex: {student.sex}")
    print(f"  Study time: {student.studytime}/4, Failures: {student.failures}, Absences: {student.absences}")
    print(f"  Social (goout): {student.goout}/5, Alcohol Weekday (Dalc): {student.Dalc}/5, Weekend (Walc): {student.Walc}/5")
    print(f"  Family support: {student.famsup}, Parents status: {student.Pstatus}, Family relations: {student.famrel}/5")
    print(f"  Higher education goal: {student.higher}, Internet: {student.internet}, Paid tutoring: {student.paid}")
    print(f"  Health: {student.health}/5, Travel time: {student.traveltime}/4")
    print(f"  Grades: G1={student.G1}, G2={student.G2}, G3={student.G3}")


def print_prediction_results(results: Dict, student: StudentProfile):
    print("\n📊 PREDICTION RESULTS")
    print("=" * 60)
    emoji = "✅" if results['prediction'] == 'PASS' else "❌"
    print(f"🎯 Prediction:       {emoji} {results['prediction']}")
    print(f"📈 Pass Probability: {results['pass_probability']:.1f}%")
    tier = results.get('confidence_tier', '')
    note = results.get('confidence_note', '')
    te = {'HIGH CONFIDENCE PASS':'🟢','MODERATE CONFIDENCE PASS':'🟡','BORDERLINE PASS':'🟠','BORDERLINE FAIL':'🟠','MODERATE CONFIDENCE FAIL':'🔴','HIGH CONFIDENCE FAIL':'💀','LIKELY PASS (Predicted)':'🟢','AT RISK (Predicted)':'🟡','HIGH RISK (Predicted)':'🔴'}.get(tier, 'ℹ️')
    print(f"🎖️  Confidence:       {te} {tier}")
    print(f"📌 Grade Note:       {note}")
    print(f"📋 Rules Applied:    {results['rules_applied']}")
    print(f"\n📝 Grades: G1={student.G1}, G2={student.G2}, G3={student.G3}")

    # Triple failure block
    if results.get('triple_failure'):
        print("\n⚠️⚠️⚠️  TRIPLE FAILURE DETECTED  ⚠️⚠️⚠️")
        print("   Student failed ALL THREE tests (G1, G2, G3)")
        print("   MUST REPEAT THE COURSE")
        print("\n📋 Past Mistakes:")
        for name, grade in results['failed_tests']:
            print(f"  • {name}: {grade}/20 - FAILED")

    # Critical factors
    if results['critical_factors']:
        print(f"\n🔴 Critical Risk Factors ({len(results['critical_factors'])}):")
        for f in results['critical_factors']:
            print(f"  • {f}")

    # Moderate factors
    if results['moderate_factors']:
        print(f"\n🟡 Moderate Risk Factors ({len(results['moderate_factors'])}):")
        for f in results['moderate_factors']:
            print(f"  • {f}")

    # Positive factors
    if results['positive_factors']:
        print(f"\n🟢 Positive Factors ({len(results['positive_factors'])}):")
        for f in results['positive_factors']:
            print(f"  • {f}")

    # Insights
    if results.get('insights'):
        print(f"\n💡 Insights ({len(results['insights'])}):")
        for ins in results['insights']:
            print(f"  • {ins}")


def print_intervention_results(results: Dict):
    print("\n🤖 INTERVENTION SEARCH RESULTS")
    print("=" * 60)
    status_map = {
        'PASS': '✅ PASS',
        'CRITICAL_FAIL': '⚠️  CRITICAL FAIL',
        'INTERVENTION_FOUND': '🔍 FAIL — Intervention Available',
        'CRITICAL': '❌ FAIL — Critical Case'
    }
    print(f"📊 Current Status: {status_map.get(results['status'], results['status'])}")
    print(f"💭 Message: {results['message']}")

    if results['steps']:
        print(f"\n📋 Recommended Steps ({len(results['steps'])}):")
        for i, step in enumerate(results['steps'], 1):
            print(f"  {i}. {step}")

    print(f"\n🎯 Final Prediction: {results['final_prediction']}")
    if results['nodes_explored'] > 0:
        print(f"🔍 Nodes Explored: {results['nodes_explored']}")


def print_triple_failure_detail(student: StudentProfile):
    print("\n" + "⚠️" * 20)
    print("⚠️⚠️⚠️  CRITICAL: YOU MUST REPEAT THIS COURSE  ⚠️⚠️⚠️")
    print("⚠️" * 20)
    print("\nYou have failed ALL THREE tests:")
    print(f"  1. G1 (First Period):  {student.G1}/20 - FAILED")
    print(f"  2. G2 (Second Period): {student.G2}/20 - FAILED")
    print(f"  3. G3 (Final Grade):   {student.G3}/20 - FAILED")
    print("\n📊 Your Grades:")
    print(f"  • G1 (First Period):  {student.G1}/20")
    print(f"  • G2 (Second Period): {student.G2}/20")
    print(f"  • G3 (Final Grade):   {student.G3}/20")
    print("\n❌ Passing Grade Required: 10/20")
    print("\n🎯 What You Need to Do:")
    print("  1. Score ≥ 10/20 on ALL tests in the next attempt")
    print(f"  2. Improve study habits (currently {student.studytime}/4 — target: 3+)")
    print(f"  3. Reduce absences (currently {student.absences} days — target: under 10)")
    print("  4. Attend tutoring or extra academic help sessions")
    print(f"  5. Reduce social activities (currently {student.goout}/5 — target: under 3)")
    print(f"  6. Reduce alcohol consumption (Dalc={student.Dalc}/5 — target: 1)")
    print("\n💡 Past Mistakes to Avoid:")
    print(f"  • G1 (First Period):  {student.G1}/20 - FAILED")
    print(f"  • G2 (Second Period): {student.G2}/20 - FAILED")
    print(f"  • G3 (Final Grade):   {student.G3}/20 - FAILED")
    print("\n⏰ Time to Repeat: Next Semester")


# ══════════════════════════════════════════════════════════════════════════════
# 8. LIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════

def run_live_demo():
    system = EduPredictAI()

    print("\n" + "=" * 60)
    print("🎓  EduPredict AI — Live Demo")
    print("🧠  Classical AI: Rule-Based + A* Search + Forward Chaining")
    print("=" * 60)

    while True:
        print("\n📌 Select Test Case:")
        print("  1. Generate Random Student")
        print("  2. Test Triple Failure Student (G1=5, G2=4, G3=3)")
        print("  3. Test Failing Student (G1=12, G2=8, G3=7)")
        print("  4. Test Passing Student (G1=12, G2=11, G3=10)")
        print("  q. Quit")
        choice = input("\nEnter choice: ").strip().lower()

        if choice == 'q':
            print("\n👋 Goodbye!")
            break
        elif choice == '1':
            student = generate_random_student()
        elif choice == '2':
            student = create_triple_failure_student()
        elif choice == '3':
            student = create_failing_student()
        elif choice == '4':
            student = create_sample_student()
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, 4, or q.")
            continue

        print_student_profile(student)

        results = system.predict(student)
        print_prediction_results(results, student)

        if student.has_triple_failure():
            print_triple_failure_detail(student)

        intervention = system.find_intervention(student)
        print_intervention_results(intervention)

        print("\n" + "-" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    run_live_demo()