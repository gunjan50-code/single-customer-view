"""
Curated reference data for generating realistic Indian customer records.

Why hand-curated instead of pure Faker: the corruption engine needs *controlled*
variants to work. If we want to test whether the matcher can connect
"Bengaluru" to "Blore", we need to know that mapping exists in the first place.
Faker gives us random strings; this gives us random strings with known,
realistic failure modes baked in.
"""

FIRST_NAMES_MALE = [
    "Rajesh", "Amit", "Suresh", "Vikram", "Arjun", "Ravi", "Sanjay", "Anil",
    "Deepak", "Manoj", "Nikhil", "Rahul", "Karthik", "Praveen", "Sandeep",
    "Ashok", "Vinod", "Gaurav", "Harish", "Naveen", "Prakash", "Sunil",
    "Vivek", "Abhishek", "Rohit", "Siddharth", "Aditya", "Mohit", "Varun",
    "Ganesh", "Ramesh", "Mahesh", "Dinesh", "Kiran", "Sachin", "Nitin",
]

FIRST_NAMES_FEMALE = [
    "Priya", "Sunita", "Anjali", "Kavita", "Meera", "Divya", "Pooja", "Neha",
    "Swati", "Lakshmi", "Rekha", "Shalini", "Nandini", "Aishwarya", "Sneha",
    "Ritu", "Deepa", "Anita", "Vandana", "Preeti", "Manisha", "Shweta",
    "Bhavana", "Radhika", "Sushma", "Jyoti", "Kalpana", "Namrata", "Sarita",
]

MIDDLE_NAMES = [
    "Kumar", "Prasad", "Chandra", "Nath", "Lal", "Devi", "Rani", "Bai",
    "Krishna", "Mohan", "Raj", "Kant", "Bhushan", "Shankar", "", "", "",
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Patel", "Reddy", "Nair", "Iyer", "Menon",
    "Desai", "Joshi", "Kulkarni", "Bhat", "Rao", "Naidu", "Pillai", "Shetty",
    "Chauhan", "Mehta", "Agarwal", "Bansal", "Sinha", "Mishra", "Tiwari",
    "Pandey", "Chatterjee", "Banerjee", "Mukherjee", "Das", "Ghosh", "Bose",
    "Malhotra", "Kapoor", "Khanna", "Chopra", "Saxena", "Srivastava",
]

# Common short forms people actually use for themselves. The support system in
# particular tends to record whatever the customer said on the phone.
NICKNAMES = {
    "Rajesh": "Raju", "Suresh": "Suri", "Ramesh": "Ramu", "Mahesh": "Mahi",
    "Rahul": "Rahu", "Abhishek": "Abhi", "Siddharth": "Sid", "Aditya": "Adi",
    "Praveen": "Pravi", "Sandeep": "Sandy", "Harish": "Hari", "Naveen": "Navi",
    "Ganesh": "Ganu", "Sachin": "Sachu", "Karthik": "Karthi", "Vikram": "Vicky",
    "Priya": "Pri", "Sunita": "Sunny", "Anjali": "Anju", "Kavita": "Kavi",
    "Divya": "Divi", "Pooja": "Puja", "Lakshmi": "Lachu", "Aishwarya": "Aish",
    "Deepa": "Dee", "Manisha": "Mannu", "Radhika": "Radha", "Namrata": "Nammu",
}

# The single most realistic source of "same person looks different" in Indian
# customer data: cities have official names, colonial names, and local slang,
# and all three end up in production databases.
CITY_VARIANTS = {
    "Bengaluru":          ["Bangalore", "Blore", "Bengaluru", "BLR"],
    "Mumbai":             ["Bombay", "Mumbai", "Mum"],
    "Kolkata":            ["Calcutta", "Kolkata", "Cal"],
    "Chennai":            ["Madras", "Chennai"],
    "Pune":               ["Poona", "Pune"],
    "Hyderabad":          ["Hyderabad", "Hyd", "Secunderabad"],
    "New Delhi":          ["Delhi", "New Delhi", "N Delhi", "NCR"],
    "Ahmedabad":          ["Amdavad", "Ahmedabad", "A'bad"],
    "Gurugram":           ["Gurgaon", "Gurugram"],
    "Kochi":              ["Cochin", "Kochi"],
    "Mysuru":             ["Mysore", "Mysuru"],
    "Vadodara":           ["Baroda", "Vadodara"],
    "Thiruvananthapuram": ["Trivandrum", "Thiruvananthapuram", "TVM"],
    "Puducherry":         ["Pondicherry", "Puducherry", "Pondy"],
    "Visakhapatnam":      ["Vizag", "Visakhapatnam"],
    "Indore":             ["Indore"],
    "Jaipur":             ["Jaipur"],
    "Lucknow":            ["Lucknow"],
    "Nagpur":             ["Nagpur"],
    "Coimbatore":         ["Coimbatore", "Kovai"],
}

# Reverse lookup built once: any variant -> the canonical city name.
# src/standardize.py uses this to collapse all spellings back together.
CITY_CANONICAL = {
    variant.lower(): canonical
    for canonical, variants in CITY_VARIANTS.items()
    for variant in variants
}

CITY_TO_STATE = {
    "Bengaluru": "Karnataka", "Mysuru": "Karnataka", "Mumbai": "Maharashtra",
    "Pune": "Maharashtra", "Nagpur": "Maharashtra", "Kolkata": "West Bengal",
    "Chennai": "Tamil Nadu", "Coimbatore": "Tamil Nadu", "Hyderabad": "Telangana",
    "New Delhi": "Delhi", "Gurugram": "Haryana", "Ahmedabad": "Gujarat",
    "Vadodara": "Gujarat", "Kochi": "Kerala", "Thiruvananthapuram": "Kerala",
    "Puducherry": "Puducherry", "Visakhapatnam": "Andhra Pradesh",
    "Indore": "Madhya Pradesh", "Jaipur": "Rajasthan", "Lucknow": "Uttar Pradesh",
}

STREET_NAMES = [
    "MG Road", "Brigade Road", "Church Street", "Residency Road", "Anna Salai",
    "Linking Road", "Park Street", "Nehru Road", "Gandhi Nagar", "Station Road",
    "Ring Road", "Main Road", "Cross Road", "Temple Street", "Market Road",
    "Hospital Road", "College Road", "Lake View Road", "Palm Grove Avenue",
    "Rose Garden Lane", "Sector 14", "Phase 2", "Jubilee Hills", "Banjara Hills",
]

# Street-type abbreviations that different systems apply inconsistently.
# standardize.py expands all of these back to the long form.
ADDRESS_ABBREVIATIONS = {
    "Road": "Rd", "Street": "St", "Avenue": "Ave", "Lane": "Ln",
    "Nagar": "Ngr", "Cross": "Crs", "Main": "Mn", "Sector": "Sec",
    "Phase": "Ph", "Apartment": "Apt", "Building": "Bldg", "Floor": "Flr",
}

ADDRESS_EXPANSIONS = {
    short.lower(): long for long, short in ADDRESS_ABBREVIATIONS.items()
}
ADDRESS_EXPANSIONS.update({
    "rd.": "road", "st.": "street", "ave.": "avenue", "ln.": "lane",
    "no.": "number", "#": "number", "opp": "opposite", "nr": "near",
    "apt.": "apartment", "bldg.": "building", "flr.": "floor",
})

EMAIL_DOMAINS = [
    "gmail.com", "yahoo.co.in", "outlook.com", "rediffmail.com",
    "hotmail.com", "gmail.com", "gmail.com",  # weighted: gmail is dominant
]

# Physical keyboard layout, used to generate *believable* typos.
# A human mistyping "Sharma" hits a key next to the one they meant, so we
# produce "Sharna" rather than a random character swap.
KEYBOARD_NEIGHBOURS = {
    "q": "wa",  "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
    "a": "qsz", "s": "awdx", "d": "sefc", "f": "drgv", "g": "fthb",
    "h": "gyjn", "j": "hukm", "k": "jil",  "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}
