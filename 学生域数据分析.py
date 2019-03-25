# 对学生基本信息进行画像展示分析。包括性别、年级、班级、住址、班主任等形成学生画像标签。
# 对个体维度对学生学业情况进行描述性统计分析。
# 对成绩情况进行统计，并汇总各个科目历史考试成绩趋势，
# 明确学生当前学科成绩分布特点以及未来成绩趋势，为学业干预提供输入。
# 度量指标如原始分、得分率、标准分（Z以及T分）、全年级排名、全班排名、离均值等。
# 学生消费画像分析，通过对学生一卡通消费数据，分析学生消费情况；
# 支持消费分布数据统计分析；
# 如消费趋势对比，对消费进行预警，便于了解学生生活方式尤其是贫困生，并及时干预支持消费明细的查询。
# 学生考勤画像分析，如学生考勤数据统计：如缺勤、迟到、请假、到勤的比例和实际天数；
import pandas as pd
from ClassDef import *
# 数据加载与查看
path = './data/'
StuInfoCol = ['ID', 'name', 'sex', 'nation', 'borndate',
       'classname', 'addr', 'home', 'policy', 'classID',
       'term', 'isInSchool', 'isLeaved', 'qinshi']
stuInfo = pd.read_csv(path+'/2_student_info.csv', names=StuInfoCol,encoding='utf8')
# 学生成绩
stuScore = pd.read_csv(path+'/5_chengji.csv')
#

# 学生基本画像展示分析，性别，年级，班级。住址，班主任
stuIds = stuInfo['ID'].values
stuName = stuInfo['name'].values
stuSex = stuInfo['sex'].values
stuNation = stuInfo['nation'].values
stuBorn = stuInfo['borndate'].values
stuClaName = stuInfo['classname'].values
stuAddr = stuInfo['addr'].values
stuHome = stuInfo['home'].values
stuPolicy = stuInfo['policy']
stuClaID = stuInfo['classID'].values
stuTerm = stuInfo['term']
stuIsIn = stuInfo['isInSchool'].values
stuIsLeaved = stuInfo['isLeaved'].values
stuQinshi = stuInfo['qinshi'].values

students = []
# 创建学生类,name='', stuId=0,sex='', grade=0, cla=0, addr='', headTeacher=''
for i in range(len(stuIds)):
    student = Student()
    student.stuId = stuIds[i]
    student.name = stuName[i]
    student.sex = stuSex[i]
    student.cla = stuClaID[i]
    students.append(student)
# 定义学生性别0：女，1：男
for student in students:
    if student.sex == 'nv':
        student.sex = 0
    else:
        student.sex == 1



teacherDF = pd.read_csv(path+'/1_teacher.csv')

teachTerm = teacherDF['term'].values
teachClaID = teacherDF['cla_id'].values
teachClaName = teacherDF['cla_Name'].values
teachGraName = teacherDF['gra_Name'].values
teachSubID = teacherDF['sub_id'].values
teachsubName = teacherDF['sub_Name'].values
teachBasID = teacherDF['bas_id'].values
teachBasName = teacherDF['bas_Name'].values
teachers = []

# 创建学生类,name='', stuId=0,sex='', grade=0, cla=0, addr='', headTeacher=''
for i in range(len(stuIds)):
    teacher = Teacher()
    teacher.term = teachTerm[i]
    teacher.claID = teachClaID[i]
    teacher.claName = teachClaName[i]
    teacher.graName = teachGraName[i]
    teacher.subID = teachSubID[i]
    teacher.subName = teachsubName[i]
    teacher.basID = teachBasID[i]
    teacher.basName = teachBasName[i]
    teachers.append(teacher)

for i in range(len())






