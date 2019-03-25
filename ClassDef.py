# 对学生基本信息进行画像展示分析。包括性别、年级、班级、住址、班主任等形成学生画像标签。
# 对个体维度对学生学业情况进行描述性统计分析。
# 对成绩情况进行统计，并汇总各个科目历史考试成绩趋势，
# 明确学生当前学科成绩分布特点以及未来成绩趋势，为学业干预提供输入。
# 度量指标如原始分、得分率、标准分（Z以及T分）、全年级排名、全班排名、离均值等。
# 学生消费画像分析，通过对学生一卡通消费数据，分析学生消费情况；
# 支持消费分布数据统计分析；
# 如消费趋势对比，对消费进行预警，便于了解学生生活方式尤其是贫困生，并及时干预支持消费明细的查询。
# 学生考勤画像分析，如学生考勤数据统计：如缺勤、迟到、请假、到勤的比例和实际天数；

class Student:
    def __init__(self,name='', stuId=0,sex='', grade=0, cla=0, addr='', headTeacher='', scores=[]):
        self.stuId = stuId
        self.name = name
        self.sex = sex
        self.grade = grade
        self.cla = cla
        self.addr = addr
        self.headTeacher = headTeacher
        self.scores = scores

class Teacher:
    """
    1_teacher.csv:包含了近五年各班各学科的教师信息
    term:学期
    cla_id:班级ID
    cla_Name:班级名
    gra_Name:年级名
    sub_id:学科ID
    sub_Name:学科名
    bas_id:教师id
    bas_Name:教师名
    """
    def __init__(self,term, claID, claName, graName, subID, subName, basID, basName):
        self.term = term
        self.claID = claID
        self.claName = claName
        self.graName = graName
        self.subID = subID
        self.subName = subName
        self.basID = basID
        self.basName = basName

class Score:
    """
    mes_TestID,考试id
    exam_number,考试编码
    exam_numname,考试编码名称
    mes_sub_id,考试学科id
    mes_sub_name,考试学科名
    exam_term,考试学期
    exam_type,考试类型（对应考试类型表）
    exam_sdate,考试开始时间
    mes_StudentID,学生id
    mes_Score,考试成绩(-1为作弊，-2为缺考，-3为免考)
    mes_Z_Score,换算成Z-score
    mes_T_Score,换算成T-score
    mes_dengdi：换算成等第
    """
    def __init__(self,TestID, exam_number, exam_numname, sub_id,
       sub_name, exam_term, exam_type, exam_sdate, StudentID,
       score, Z_score, T_score, dengdi):
        self.TestID = TestID
        self.exam_number = exam_number
        self.exam_numname = exam_numname
        self.sub_id =sub_id
        self.sub_name = sub_name
        self.exam_term = exam_term
        self.exam_type = exam_type
        self.exam_sdate = exam_sdate
        self.studentID = StudentID
        self.score = score
        self.T_score = T_score
        self.Z_score = Z_score
        self.dengdi = dengdi

# 班级

