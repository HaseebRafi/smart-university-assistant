"""
generate_faq_data.py
---------------------
Creates data/university_faq.csv with 100+ manually-written Q&A pairs across
the 10 required categories. Replace/extend the content below with your own
institution's real policies before final submission (rule #3 in the
assignment: chatbot responses must be grounded in your FAQ dataset).
"""
import csv
import os

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "data", "university_faq.csv")

FAQS = [
    # (category, question, answer, keywords, source)
    ("Admissions", "How can I apply for admission?", "Applications can be submitted through the university admission portal before the published deadline.", "admission,apply,portal", "Admission Policy"),
    ("Admissions", "What documents are required for admission?", "Applicants must upload their transcripts, a copy of their national ID or passport, and passport-size photographs.", "documents,admission,transcripts", "Admission Policy"),
    ("Admissions", "Is there an admission test?", "Most undergraduate programs require an entry test, while some postgraduate programs require an interview instead.", "test,interview,admission", "Admission Policy"),
    ("Admissions", "What is the last date to apply?", "Admission deadlines are published on the university website at the start of each intake and vary by program.", "deadline,apply,intake", "Admission Policy"),
    ("Admissions", "Can I apply for more than one program?", "Applicants may apply for up to two programs in the same intake, subject to separate application fees.", "multiple programs,apply", "Admission Policy"),
    ("Admissions", "How do I check my admission status?", "Admission status can be checked by logging into the applicant portal with your application ID.", "status,portal,application", "Admission Policy"),
    ("Admissions", "Is admission merit-based?", "Admission is offered based on a combined merit score of academic record and entry test performance.", "merit,admission,score", "Admission Policy"),
    ("Admissions", "Can international students apply?", "International students may apply through the international admissions office and must submit equivalence certificates.", "international,equivalence", "Admission Policy"),
    ("Admissions", "What is the minimum eligibility criteria?", "Applicants must have a minimum of 50 percent marks in their previous qualification to be eligible.", "eligibility,criteria,marks", "Admission Policy"),
    ("Admissions", "Can I defer my admission to next semester?", "Deferral requests are considered on a case-by-case basis and must be submitted before the semester start.", "defer,admission,semester", "Admission Policy"),
    ("Admissions", "How do I get a transcript evaluation?", "Transcript evaluation is done by the admissions office once official transcripts are received from the previous institution.", "transcript,evaluation", "Admission Policy"),

    ("Fee and Finance", "Can I pay my fee in instalments?", "Instalment requests are subject to approval from the finance office and are usually split into two payments per semester.", "fee,instalment,finance", "Fee Policy"),
    ("Fee and Finance", "What happens if I miss the fee deadline?", "A late fee surcharge is applied after the due date, and access to the learning portal may be suspended.", "late fee,deadline", "Fee Policy"),
    ("Fee and Finance", "How can I pay my tuition fee?", "Tuition fees can be paid online through the student portal, via bank transfer, or at designated bank branches.", "pay,tuition,bank", "Fee Policy"),
    ("Fee and Finance", "Is the admission fee refundable?", "The admission fee is non-refundable once the seat has been confirmed and the semester has started.", "refund,admission fee", "Fee Policy"),
    ("Fee and Finance", "How do I get a fee receipt?", "Fee receipts are automatically generated and available for download from the student finance dashboard.", "receipt,fee", "Fee Policy"),
    ("Fee and Finance", "Are there fee concessions for siblings?", "A sibling discount is available when two or more siblings are enrolled at the university at the same time.", "sibling,discount,concession", "Fee Policy"),
    ("Fee and Finance", "What is the semester fee structure?", "Semester fees include tuition, examination, and library charges, and vary by program and credit hours.", "fee structure,semester", "Fee Policy"),
    ("Fee and Finance", "Can I get a fee waiver for financial hardship?", "Students facing financial hardship may apply for a need-based fee waiver through the student affairs office.", "waiver,hardship,fee", "Fee Policy"),
    ("Fee and Finance", "Will I be fined for late fee payment?", "A fixed late payment fine is added for each week the payment remains overdue after the due date.", "fine,late payment", "Fee Policy"),
    ("Fee and Finance", "How do I request a fee refund after withdrawal?", "Refund requests after withdrawal are processed according to the refund schedule published each semester.", "refund,withdrawal,fee", "Fee Policy"),
    

    ("Course Registration", "How can I register for courses?", "Course registration is completed through the student portal during the registration window announced each semester.", "registration,course,portal", "Registration Guide"),
    ("Course Registration", "Can I change my registered courses?", "Courses can be added or dropped during the add/drop period at the start of the semester.", "add,drop,courses", "Registration Guide"),
    ("Course Registration", "What is the maximum number of credit hours per semester?", "Students may normally register for up to 18 credit hours per semester without special permission.", "credit hours,limit", "Registration Guide"),
    ("Course Registration", "What happens if I miss course registration?", "Late registration is allowed for a short grace period with a late registration fee.", "late registration,fee", "Registration Guide"),
    ("Course Registration", "How do I register for a repeat course?", "Repeat courses must be registered manually with approval from the academic advisor.", "repeat,course,advisor", "Registration Guide"),
    ("Course Registration", "Can I take an overload of courses?", "An overload beyond the normal credit limit requires approval based on your cumulative GPA.", "overload,gpa,credit hours", "Registration Guide"),
    ("Course Registration", "How do I know if I meet a course prerequisite?", "Prerequisites for each course are listed in the course catalogue and checked automatically during registration.", "prerequisite,course catalogue", "Registration Guide"),
    ("Course Registration", "Can I register for an elective from another department?", "Cross-department electives are allowed subject to seat availability and departmental approval.", "elective,department", "Registration Guide"),
    ("Course Registration", "What is a waitlist in course registration?", "If a course section is full, students are placed on a waitlist and enrolled automatically if a seat opens.", "waitlist,section,full", "Registration Guide"),
    ("Course Registration", "How do I withdraw from a single course?", "A single course can be withdrawn before the withdrawal deadline without affecting other registered courses.", "withdraw,course", "Registration Guide"),

    ("Attendance", "What is the minimum attendance requirement?", "Students must maintain the attendance percentage stated in university regulations, normally 75 percent per course.", "attendance,requirement", "Attendance Policy"),
    ("Attendance", "What happens if my attendance is below the requirement?", "Students below the minimum attendance threshold may be barred from sitting the final examination.", "attendance,barred,exam", "Attendance Policy"),
    ("Attendance", "Can I get attendance relaxation for medical reasons?", "Medical leave supported by a valid certificate can be considered for attendance relaxation on a case-by-case basis.", "medical,relaxation,attendance", "Attendance Policy"),
    ("Attendance", "How is attendance calculated?", "Attendance is calculated as the percentage of classes attended out of the total classes held for each course.", "attendance,calculation", "Attendance Policy"),
    ("Attendance", "Where can I check my attendance record?", "Attendance records are visible to students in real time through the learning management system.", "attendance record,portal", "Attendance Policy"),
    ("Attendance", "Does online class attendance count?", "Attendance in scheduled online sessions is recorded the same way as in-person attendance.", "online,attendance", "Attendance Policy"),
    ("Attendance", "Can attendance shortage be waived by the instructor?", "Attendance shortage waivers must be approved by the department, not by an individual instructor alone.", "waiver,shortage,department", "Attendance Policy"),
    ("Attendance", "Is attendance counted for university-approved events?", "Absences for official university-approved events such as sports or competitions are marked as excused.", "excused,events,attendance", "Attendance Policy"),

    ("Examinations", "What is the minimum passing percentage?", "Students must obtain at least 50 percent marks in a course to pass, unless stated otherwise for a specific program.", "pass,marks,percentage", "Examination Policy"),
    ("Examinations", "What happens if I miss an exam?", "Students who miss an exam due to a valid emergency must apply for a make-up exam within the specified period.", "missed exam,make-up", "Examination Policy"),
    ("Examinations", "How can I apply for a re-check of my exam paper?", "Rechecking requests can be submitted through the examination office within one week of result announcement.", "recheck,exam paper", "Examination Policy"),
    ("Examinations", "When are final exam results announced?", "Final results are typically announced within three to four weeks after the completion of examinations.", "results,announcement", "Examination Policy"),
    ("Examinations", "What is considered examination misconduct?", "Copying, using unauthorized material, or impersonation during an exam are all treated as examination misconduct.", "misconduct,cheating,exam", "Examination Policy"),
    ("Examinations", "Can I retake an exam to improve my grade?", "Grade-improvement exams are allowed for select courses under the improvement policy, subject to a fee.", "retake,improvement,grade", "Examination Policy"),
    ("Examinations", "Are calculators allowed in exams?", "Calculators are allowed only where explicitly stated on the exam paper or by the course instructor.", "calculator,exam rules", "Examination Policy"),
    ("Examinations", "How many hours is a typical final exam?", "Final examinations are typically two to three hours long depending on the course credit hours.", "exam duration,hours", "Examination Policy"),
    ("Examinations", "What if I disagree with my final grade?", "Grade appeals can be filed with the controller of examinations within the appeal window after result declaration.", "grade appeal,dispute", "Examination Policy"),

    ("Scholarships", "How do I apply for a scholarship?", "Scholarship applications are submitted through the financial aid office along with supporting income documents.", "scholarship,apply,financial aid", "Scholarship Policy"),
    ("Scholarships", "What types of scholarships are available?", "The university offers merit scholarships, need-based scholarships, and sports scholarships.", "merit,need-based,sports", "Scholarship Policy"),
    ("Scholarships", "What GPA is required to maintain a merit scholarship?", "Merit scholarship holders must maintain a minimum cumulative GPA as specified in the scholarship award letter.", "gpa,merit,maintain", "Scholarship Policy"),
    ("Scholarships", "Can I apply for a scholarship after the first semester?", "Continuing students can apply for scholarships in subsequent semesters if they meet the eligibility criteria.", "continuing,apply,scholarship", "Scholarship Policy"),
    ("Scholarships", "Are scholarships available for postgraduate students?", "A limited number of research-based scholarships are available for postgraduate and doctoral students.", "postgraduate,research,scholarship", "Scholarship Policy"),
    ("Scholarships", "What happens if my grades fall below the scholarship requirement?", "Scholarships are suspended for one semester if the required GPA is not maintained, and cancelled after repeated shortfall.", "suspend,gpa,shortfall", "Scholarship Policy"),
    ("Scholarships", "Is there a scholarship for financially needy students?", "Need-based scholarships are awarded following a review of family income and supporting documents by the financial aid committee.", "need-based,income,review", "Scholarship Policy"),

    ("Internships", "Where can I find internship opportunities?", "Internship listings are posted by the career services office and are also shared through the student portal.", "internship,career services", "Career Services Guide"),
    ("Internships", "Is an internship mandatory for graduation?", "Most undergraduate programs require a mandatory internship of six to eight weeks before the final semester.", "mandatory,internship,graduation", "Internship Policy"),
    ("Internships", "How do I get credit for my internship?", "Internship credit is awarded after submitting a supervisor evaluation and an internship report to the department.", "internship credit,report", "Internship Policy"),
    ("Internships", "Can I do a remote internship?", "Remote internships are accepted provided the host organization confirms supervised work and issues a completion certificate.", "remote internship", "Internship Policy"),
    ("Internships", "Who approves my internship placement?", "Internship placements must be approved in advance by the internship coordinator in your department.", "approval,coordinator,internship", "Internship Policy"),
    ("Internships", "Can I arrange my own internship instead of using the university list?", "Students may arrange a self-sourced internship as long as it is approved by the internship coordinator beforehand.", "self-sourced,internship", "Internship Policy"),

    ("Final-Year Projects", "How do I choose a final-year project topic?", "Project topics are chosen from a list of proposed topics or proposed independently and approved by a supervisor.", "project topic,supervisor", "Final Year Project Guide"),
    ("Final-Year Projects", "When should I submit my final-year project proposal?", "Project proposals are typically due within the first month of the final-year project semester.", "proposal,deadline,project", "Final Year Project Guide"),
    ("Final-Year Projects", "Can I work on a final-year project in a group?", "Final-year projects can be completed individually or in groups of up to three students, depending on program rules.", "group project,team", "Final Year Project Guide"),
    ("Final-Year Projects", "Who evaluates the final-year project?", "Final-year projects are evaluated by an internal panel and, for some programs, an external examiner.", "evaluation,panel,examiner", "Final Year Project Guide"),
    ("Final-Year Projects", "What happens if my project is not completed on time?", "An incomplete grade may be issued with an extension period, after which the project must be resubmitted.", "incomplete,extension,project", "Final Year Project Guide"),
    ("Final-Year Projects", "Do I need to submit a project report?", "A formal written report following the department's format guidelines must be submitted alongside the project defense.", "report,defense,format", "Final Year Project Guide"),

    ("Library Services", "What are the library timings?", "The main library is open from 8 AM to 10 PM on weekdays and 9 AM to 5 PM on weekends.", "library timings,hours", "Library Services Guide"),
    ("Library Services", "How many books can I borrow at once?", "Undergraduate students can borrow up to five books at a time for a two-week loan period.", "borrow,books,loan period", "Library Services Guide"),
    ("Library Services", "What happens if I return a book late?", "A small daily fine is charged for each day a book is returned after the due date.", "late fine,library book", "Library Services Guide"),
    ("Library Services", "How do I access digital journals?", "Digital journals and e-books can be accessed remotely using your student ID through the library's online portal.", "digital journals,e-books", "Library Services Guide"),
    ("Library Services", "Can I reserve a study room in the library?", "Study rooms can be reserved in advance through the library booking system, subject to availability.", "study room,reserve", "Library Services Guide"),
    ("Library Services", "What do I do if I lose a borrowed book?", "A lost book must be reported to the library, and the student is charged the replacement cost of the book.", "lost book,replacement cost", "Library Services Guide"),

    ("Campus Facilities", "Is on-campus accommodation available?", "Limited hostel accommodation is available on a first-come, first-served basis for out-of-city students.", "hostel,accommodation,campus", "Campus Facilities Guide"),
    ("Campus Facilities", "Is there a medical center on campus?", "The campus medical center provides first aid and basic consultations during working hours on weekdays.", "medical center,health", "Campus Facilities Guide"),
    ("Campus Facilities", "Are sports facilities available for students?", "The campus offers a gymnasium, football ground, and indoor games facilities for enrolled students.", "sports,gym,facilities", "Campus Facilities Guide"),
    ("Campus Facilities", "Is Wi-Fi available across campus?", "Campus-wide Wi-Fi is available to all students using their university credentials.", "wifi,internet,campus", "Campus Facilities Guide"),
    ("Campus Facilities", "Where can students eat on campus?", "The campus cafeteria and several food kiosks are open throughout the day for students and staff.", "cafeteria,food,campus", "Campus Facilities Guide"),
    ("Campus Facilities", "Is parking available for students?", "Designated student parking areas are available near the main campus gates, subject to a parking permit.", "parking,permit,campus", "Campus Facilities Guide"),
    ("Campus Facilities", "Are there prayer facilities on campus?", "Prayer rooms for various faiths are available in the main academic block and student center.", "prayer room,facilities", "Campus Facilities Guide"),
    
    # A few extra cross-category entries to comfortably exceed 100 records
    ("Admissions", "Do I need to submit an SOP for admission?", "A statement of purpose is required for most postgraduate and research-based program applications.", "sop,statement of purpose", "Admission Policy"),
    ("Fee and Finance", "Is there an education loan facility?", "The university partners with select banks to offer education loan facilities to eligible students.", "education loan,bank", "Fee Policy"),
    ("Course Registration", "Can I audit a course without credit?", "Auditing a course is allowed with instructor permission but does not count toward degree credit hours.", "audit,course,credit", "Registration Guide"),
    ("Attendance", "Is attendance tracked automatically in online sessions?", "Attendance in online sessions is tracked automatically through the video conferencing platform's log.", "online attendance,tracking", "Attendance Policy"),
    ("Examinations", "Can I bring my own stationery to the exam hall?", "Students must bring their own stationery; sharing of stationery during exams is not permitted.", "stationery,exam hall", "Examination Policy"),
    ("Scholarships", "Is there a scholarship renewal process each year?", "Scholarship renewal requires submitting updated academic transcripts at the start of each academic year.", "renewal,scholarship,transcripts", "Scholarship Policy"),
    ("Internships", "Do I get a stipend during my internship?", "Stipend policies vary by host organization; the university does not guarantee a paid internship.", "stipend,internship,paid", "Internship Policy"),
    ("Final-Year Projects", "Can I change my final-year project supervisor?", "A supervisor change request can be submitted to the department within the first few weeks of the project.", "change supervisor,project", "Final Year Project Guide"),
    ("Library Services", "Can alumni access the library?", "Alumni may access library reading facilities upon registration, though borrowing privileges are limited.", "alumni,library access", "Library Services Guide"),
    ("Campus Facilities", "Is there a career counselling service on campus?", "The student affairs office provides career counselling sessions throughout the academic year.", "career counselling,student affairs", "Campus Facilities Guide"),
    ("Admissions", "Can I change my major after admission?", "A change of major is allowed within the first semester subject to meeting the new program's eligibility criteria.", "change major,program", "Admission Policy"),
    ("Fee and Finance", "Do returning students pay a re-admission fee?", "Students returning after a leave of absence pay a re-admission fee along with the applicable semester tuition.", "re-admission,leave,fee", "Fee Policy"),
    ("Course Registration", "What is the difference between a core and elective course?", "Core courses are compulsory for your degree, while electives can be chosen from a list of optional subjects.", "core course,elective", "Registration Guide"),
    ("Attendance", "Can I check my attendance for a specific course only?", "The learning management system allows filtering the attendance report by individual course.", "attendance filter,course", "Attendance Policy"),
    ("Examinations", "Is there a supplementary exam for failed courses?", "Supplementary exams are offered for failed courses in the semester immediately following the result announcement.", "supplementary exam,failed course", "Examination Policy"),
    ("Scholarships", "Can I hold two scholarships at the same time?", "Students may not hold two university-funded scholarships simultaneously unless explicitly permitted by policy.", "two scholarships,simultaneous", "Scholarship Policy"),
    ("Internships", "Does the internship office help with placement?", "The career services office assists students in finding suitable internship placements with partner organizations.", "placement,internship office", "Internship Policy"),
    ("Final-Year Projects", "Is plagiarism checked in final-year project reports?", "All final-year project reports are checked using plagiarism-detection software before final submission is accepted.", "plagiarism,project report", "Final Year Project Guide"),
    ("Library Services", "Can I renew a borrowed book online?", "Borrowed books can be renewed online through the library portal if no other student has requested them.", "renew book,online", "Library Services Guide"),
    ("Campus Facilities", "Are there printing and photocopy facilities on campus?", "Printing and photocopy kiosks are available near the library and in each academic block.", "printing,photocopy,campus", "Campus Facilities Guide"),
    # Academics & Programs Category
    ("Academics", "What types of degree programs are offered?", "NCBA&E offers Associate Degree Programs (ADP), 4-year Undergraduate degrees (BS/BBA), Graduate tracks (MBA/MPhil), and Postgraduate research (PhD).", "degrees,programs,undergraduate,graduate,phd", "Academic Profile"),
    ("Academics", "Does the university offer 2-year Bachelor degrees after an ADP?", "Yes, 2-year top-up programs like BS FinTech and BS Accounting & Finance are available for candidates with 14 years of education or a completed ADP.", "adp,top-up,bs,2 year", "Academic Profile"),
    ("Academics", "What faculties are available at the university?", "Programs are run across five schools: Business Administration, Computer Science, Social Sciences, Natural Sciences, and Arts & Humanities.", "faculties,schools,departments", "Academic Profile"),
    ("Academics", "How long does a PhD program take?", "A PhD program at NCBA&E (available in fields like Computer Science, Business Administration, Math, and Stats) takes a minimum duration of 3 years.", "phd,doctorate,duration", "Academic Profile"),

    # Admissions Category
    ("Admissions", "What is the last date to apply for the current intake?", "The deadline to submit applications for the Fall 2026 admission cycle is Wednesday, July 15, 2026.", "deadline,apply,date,fall 2026", "Admission Policy"),
    ("Admissions", "What is the eligibility requirement for a PhD program?", "Applicants must have completed 18 years of education (MPhil or equivalent) to be eligible for PhD admission.", "phd,eligibility,criteria,requirements", "Admission Policy"),
    ("Admissions", "What is the eligibility for standard undergraduate BS or BBA tracks?", "Applicants generally require a minimum of 12 years of education (Intermediate, A-Levels, or equivalent) to apply.", "undergraduate,eligibility,bs,bba,requirements", "Admission Policy"),
    ("Admissions", "How do I apply for a program?", "You can apply online by clicking the 'Apply Now' portal link on the official NCBA&E homepage.", "apply,online,portal,registration", "Admission Policy"),

    # Student Portals & Resources Category
    ("Student Portals", "What is the LMS and how do I access it?", "The Learning Management System (LMS) is the online portal where students access course catalogs, view assignments, and track timetables.", "lms,portal,login,course catalog", "IT Support"),
    ("Student Portals", "Where can I find the official university rules?", "The student Code of Conduct and the complete course catalogs are available digitally under the 'Student' tab on the website.", "rules,code of conduct,regulations", "Student Affairs"),
    ("Student Portals", "Is the university participating in the laptop scheme?", "Yes, updates and registration tracking for the CM Laptop Scheme 2026 are published directly under the website's Student Announcements section.", "laptop,cm laptop scheme,free laptop", "Student Affairs"),

    # Campuses & Contact Category
    ("Campuses", "Where are the Lahore campuses located?", "In Lahore, NCBA&E has its Main Campus, alongside specialized setups at DHA, East Canal, and FLC Lahore.", "lahore,main campus,dha,canal", "Campus Guide"),
    ("Campuses", "Does the university have campuses outside Lahore?", "Yes, NCBA&E has regional campuses operating in Multan, Bahawalpur, and Rahim Yar Khan.", "sub-campuses,multan,bahawalpur,rahim yar khan", "Campus Guide"),
    ("Campuses", "What are the official admission contact numbers?", "You can call the campus lines at +92 (42) 3575-2716/19 or contact the mobile support desk at 0302-4462223 / 0303-4462223.", "phone,contact,call,helpline", "Campus Guide"),

    # Career & Student Life Category
    ("Student Life", "Does the university help with internships and job placements?", "Yes, the CDAR (Center for Data Analysis and Research) department coordinates corporate internships, workshops, and job placements through company MoUs.", "jobs,internships,placement,cdar,mou", "Career Services"),
    ("Student Life", "What student societies can I join on campus?", "Students can join the Young Computer Professionals Society, the Management Scholar Forum, or the GotTalent Society for co-curricular activities.", "societies,clubs,extracurricular,gottalent", "Student Engagement"),
    ("Student Life", "What is the function of the QAEC department?", "The Quality Assurance & Excellence Cell (QAEC) monitors academic structures and hosts campus events like Quality Day to maintain institutional standards.", "qaec,qec,quality day,standards", "Quality Assurance"),
]

with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["faq_id", "category", "question", "answer", "keywords", "source"])
    for i, (cat, q, a, kw, src) in enumerate(FAQS, start=1):
        writer.writerow([f"FAQ{i:03d}", cat, q, a, kw, src])

print(f"Wrote {len(FAQS)} FAQ records to {OUT}")
print("Categories covered:", sorted(set(r[0] for r in FAQS)))
