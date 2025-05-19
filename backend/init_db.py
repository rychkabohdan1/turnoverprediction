import asyncio
from datetime import datetime, timedelta
import random
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
import sys

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Test data
FIRST_NAMES = [
    "John", "Emma", "Michael", "Sophia", "William", "Olivia", "James", "Ava", 
    "Alexander", "Isabella", "Benjamin", "Mia", "Elijah", "Charlotte", "Lucas",
    "Amelia", "Mason", "Harper", "Logan", "Evelyn", "Jacob", "Abigail", "David",
    "Emily", "Joseph", "Elizabeth", "Carter", "Sofia", "Owen", "Avery",
    "Daniel", "Grace", "Matthew", "Ella", "Henry", "Victoria", "Jackson", "Chloe",
    "Sebastian", "Lily", "Aiden", "Hannah", "Ethan", "Addison", "Samuel", "Natalie"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen",
    "Hill", "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera"
]

DEPARTMENTS = ["Engineering", "Sales", "Marketing", "HR", "Finance", "IT", "Operations"]
POSITIONS = {
    "Engineering": ["Software Engineer", "Senior Developer", "Tech Lead", "QA Engineer", "DevOps Engineer", "Frontend Developer", "Backend Developer"],
    "Sales": ["Sales Representative", "Sales Manager", "Account Executive", "Sales Director", "Business Development Manager"],
    "Marketing": ["Marketing Specialist", "Content Writer", "Digital Marketing Manager", "SEO Specialist", "Brand Manager"],
    "HR": ["HR Specialist", "Recruiter", "HR Manager", "Benefits Coordinator", "Training Manager"],
    "Finance": ["Financial Analyst", "Accountant", "Finance Manager", "Budget Analyst", "Financial Controller"],
    "IT": ["System Administrator", "IT Support Specialist", "Network Engineer", "Security Analyst", "Database Administrator", "IT Manager"],
    "Operations": ["Operations Manager", "Supply Chain Analyst", "Logistics Coordinator", "Facilities Manager", "Operations Director"]
}

SKILLS = [
    "Python", "JavaScript", "Java", "SQL", "Project Management", "Communication", 
    "Leadership", "Data Analysis", "Problem Solving", "Team Management",
    "React", "Angular", "Node.js", "AWS", "Azure", "Docker", "Kubernetes",
    "Agile", "Scrum", "DevOps", "Machine Learning", "Blockchain", "Cloud Computing",
    "UX/UI Design", "Content Marketing", "SEO", "SEM", "Social Media", "Email Marketing",
    "Customer Relations", "Sales", "Negotiation", "Public Speaking", "Business Strategy"
]

PROJECTS = [
    "Website Redesign", "Mobile App Development", "Cloud Migration", 
    "Customer Portal", "Data Analytics Platform", "Security Audit",
    "CRM Implementation", "Sales Dashboard", "HR System Update",
    "Financial Reporting Tool", "Marketing Campaign", "Customer Satisfaction Survey",
    "Supply Chain Optimization", "Warehouse Management System", "Product Launch",
    "Employee Retention Program", "Talent Acquisition System", "Performance Management Tool"
]

email_cache = set()  # To track used emails and avoid duplicates

def generate_email(first_name, last_name):
    """Generate a professional email address based on name."""
    email_domains = ["company.com", "enterprise.com", "corporate.com", "techcorp.com", "innovate.org"]
    
    # Try to generate a unique email
    for _ in range(10):  # Try up to 10 times
        domain = random.choice(email_domains)
        # Randomly choose email format
        format_choice = random.randint(1, 5)
        
        if format_choice == 1:
            email = f"{first_name.lower()}.{last_name.lower()}@{domain}"
        elif format_choice == 2:
            email = f"{first_name[0].lower()}{last_name.lower()}@{domain}"
        elif format_choice == 3:
            email = f"{first_name.lower()}_{last_name.lower()}@{domain}"
        elif format_choice == 4:
            email = f"{last_name.lower()}.{first_name.lower()}@{domain}"
        else:
            email = f"{first_name.lower()}{random.randint(1, 999)}@{domain}"
        
        # Check if this email is already used
        if email not in email_cache:
            email_cache.add(email)
            return email
            
    # If we couldn't generate a unique email after several tries, create one with timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    email = f"{first_name.lower()}.{last_name.lower()}.{timestamp}@{random.choice(email_domains)}"
    email_cache.add(email)
    return email

def get_performance_score():
    """Generate a more realistic performance score distribution."""
    # Higher probability of good scores (normal distribution around 3.8)
    score = random.normalvariate(3.8, 0.8)
    # Clamp between 1.0 and 5.0
    return round(max(1.0, min(5.0, score)), 1)

def get_salary_for_position(position, department):
    """Return a realistic salary range based on position and department."""
    base_salaries = {
        "Engineering": (70000, 140000),
        "Sales": (60000, 130000),
        "Marketing": (55000, 120000),
        "HR": (50000, 110000),
        "Finance": (65000, 135000),
        "IT": (60000, 130000),
        "Operations": (55000, 125000)
    }
    
    # Adjust for seniority in position name
    base_min, base_max = base_salaries.get(department, (50000, 100000))
    
    if any(senior in position.lower() for senior in ["senior", "lead", "manager", "director"]):
        return random.randint(int(base_max * 0.7), int(base_max * 1.2))
    else:
        return random.randint(base_min, int(base_max * 0.8))

async def create_test_employee(specific_department=None):
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    
    # Use specific department if provided, otherwise random
    department = specific_department if specific_department else random.choice(DEPARTMENTS)
    
    # Make sure the department exists in POSITIONS
    if department not in POSITIONS:
        print(f"Warning: Department '{department}' has no positions defined. Defaulting to Engineering.")
        department = "Engineering"
        
    position = random.choice(POSITIONS[department])
    hire_date = datetime.now() - timedelta(days=random.randint(30, 1825))  # Up to 5 years
    last_evaluation = datetime.now() - timedelta(days=random.randint(1, 180))
    
    # More realistic salary based on position and department
    salary = get_salary_for_position(position, department)
    
    # Calculate years of experience based on age and career path
    age = random.randint(22, 60)
    # Rough estimate: started working at 22, could have different careers
    max_possible_experience = age - 22
    years_of_experience = min(max_possible_experience, random.randint(1, 20))
    
    # Calculate work parameters
    work_hours = random.normalvariate(42, 5)  # Normal distribution around 42 hours
    work_hours = max(30, min(60, int(work_hours)))  # Clamp between 30-60
    
    # Project and training metrics
    projects_completed = random.randint(1, 20)
    training_hours = random.randint(5, 80)
    
    # Team size generally between 3-20
    team_size = random.randint(3, 20)
    
    # Time since last promotion (months)
    time_since_promotion = random.randint(3, 48)

    return {
        "name": f"{first_name} {last_name}",
        "first_name": first_name,
        "last_name": last_name,
        "department": department,
        "position": position,
        "risk_level": "unknown",  # Will be calculated by model
        "turnover_probability": 0.0,  # Will be calculated by model
        "email": generate_email(first_name, last_name),
        "hire_date": hire_date,
        "salary": salary,
        "performance_score": get_performance_score(),
        "last_evaluation_date": last_evaluation,
        "projects": random.sample(PROJECTS, min(random.randint(1, 5), len(PROJECTS))),
        "skills": random.sample(SKILLS, min(random.randint(2, 8), len(SKILLS))),
        "age": age,
        "years_of_experience": years_of_experience,
        "work_hours": work_hours,
        "projects_completed": projects_completed,
        "training_hours": training_hours,
        "team_size": team_size,
        "time_since_promotion": time_since_promotion
    }

async def check_users(db):
    print("\nChecking existing users:")
    users = await db.users.find().to_list(length=100)
    for user in users:
        print(f"Username: {user['username']}")
    print(f"Total users: {len(users)}")

async def check_employees(db):
    print("\nChecking existing employees:")
    count_per_department = {}
    employees = await db.employees.find().to_list(length=100)
    
    for employee in employees:
        dept = employee.get('department', 'Unknown')
        count_per_department[dept] = count_per_department.get(dept, 0) + 1
        print(f"Name: {employee['name']}, Email: {employee['email']}, Department: {employee['department']}")
    
    print(f"\nTotal employees: {len(employees)}")
    print("Department distribution:")
    for dept, count in count_per_department.items():
        print(f"  {dept}: {count} employees")

async def init_db():
    # Connect to MongoDB
    MONGODB_URL = "mongodb://admin:password123@localhost:27017"
    print(f"Connecting to MongoDB at {MONGODB_URL}")
    
    try:
        client = AsyncIOMotorClient(MONGODB_URL)
        # Verify connection
        await client.admin.command('ping')
        print("Connected to MongoDB successfully")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return
        
    db = client.turnover_predictor

    # Check existing users before creating new ones
    await check_users(db)

    # Create default admin user if not exists
    admin_user = await db.users.find_one({"username": "admin"})
    if not admin_user:
        hashed_password = pwd_context.hash("admin123")
        await db.users.insert_one({
            "username": "admin",
            "password": hashed_password
        })
        print("Default admin user created")
    else:
        print("Admin user already exists")

    # Create HR user if not exists
    hr_user = await db.users.find_one({"username": "hr"})
    if not hr_user:
        hashed_password = pwd_context.hash("hr123")
        await db.users.insert_one({
            "username": "hr",
            "password": hashed_password
        })
        print("HR user created")
    else:
        print("HR user already exists")

    # Create indexes
    await db.users.create_index("username", unique=True)
    await db.employees.create_index("email", unique=True)
    
    # Check existing employees
    await check_employees(db)
    
    # Create test employees if none exist or force creation
    existing_employees = await db.employees.count_documents({})
    force_create = "--force" in sys.argv
    
    if existing_employees == 0 or force_create:
        if force_create and existing_employees > 0:
            print(f"\nForce flag detected. Dropping {existing_employees} existing employees...")
            await db.employees.delete_many({})
            
        print("\nCreating test employees...")
        
        # Create employees with balanced distribution across departments
        target_per_department = 5  # Aim for 5 employees per department
        total_created = 0
        insert_errors = 0
        
        for department in DEPARTMENTS:
            print(f"Creating employees for {department} department...")
            created_for_dept = 0
            
            # Try to create employees for this department
            for _ in range(target_per_department):
                try:
                    # Create employee for this specific department
                    employee = await create_test_employee(department)
                    result = await db.employees.insert_one(employee)
                    if result.inserted_id:
                        created_for_dept += 1
                        total_created += 1
                        print(f"  Created: {employee['name']} ({employee['email']})")
                    else:
                        print(f"  Failed to create employee (no error, but no ID returned)")
                        insert_errors += 1
                except Exception as e:
                    print(f"  Error creating employee: {str(e)}")
                    insert_errors += 1
                    
            print(f"Created {created_for_dept} employees for {department} department")
            
        print(f"Employee creation completed. Total created: {total_created}, Errors: {insert_errors}")
    else:
        print(f"\nSkipping employee creation - {existing_employees} employees already exist.")
        print("Use --force flag to recreate employees.")
    
    # Check users and employees after initialization
    await check_users(db)
    await check_employees(db)
    
    print("Database initialization completed")

if __name__ == "__main__":
    print(f"Starting database initialization script...")
    asyncio.run(init_db()) 