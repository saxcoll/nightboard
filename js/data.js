export const event = {
  name: "Cursor Calgary Meetup",
  month: "August",
  venue: "ZayZoon",
  address: "685 Centre St S, Suite 1900, Calgary",
  timezone: "America/Edmonton",
  start: "2026-08-26T17:30:00-06:00",
  end: "2026-08-26T20:30:00-06:00",
  capacity: 245,
  present: 194,
};

export const hub = {
  teams: 35,
  pool: 194,
  people: 0,
};

export const youAre = "spaceflex";

export const schedule = [
  { id: "arrive", title: "Arrivals & mingle", start: "2026-08-26T17:30:00-06:00", end: "2026-08-26T18:00:00-06:00" },
  { id: "intro", title: "Intro to Cursor", start: "2026-08-26T18:00:00-06:00", end: "2026-08-26T18:10:00-06:00" },
  { id: "talk", title: "Code to Cloud podcast", start: "2026-08-26T18:10:00-06:00", end: "2026-08-26T18:30:00-06:00" },
  { id: "build", title: "Build", start: "2026-08-26T18:30:00-06:00", end: "2026-08-26T20:00:00-06:00" },
  { id: "demo", title: "Demos & networking", start: "2026-08-26T20:00:00-06:00", end: "2026-08-26T20:30:00-06:00" },
];

export const rooms = {
  stage: { label: "Stage", x: 42, y: 6, w: 16, h: 10 },
  pizza: { label: "Pizza", x: 78, y: 8, w: 16, h: 8 },
  checkin: { label: "Check-in", x: 6, y: 8, w: 16, h: 8 },
};

export const pizza = {
  lowAt: 3,
  trays: [
    { id: "pepperoni", name: "Pepperoni", hue: [0, 28], color: "#9b2d1f", slices: 16, max: 16 },
    { id: "cheese", name: "Cheese", hue: [38, 58], color: "#d4a017", slices: 14, max: 16 },
    { id: "veggie", name: "Veggie", hue: [85, 145], color: "#1f6b3a", slices: 11, max: 12 },
    { id: "hawaiian", name: "Hawaiian", hue: [20, 42], color: "#c45c12", slices: 6, max: 8 },
  ],
};

export const announcements = [
  { id: "a1", text: "Three teams will be picked from volunteers tonight. One click per team." },
  { id: "a2", text: "Spaceflex already volunteered to present." },
  { id: "a3", text: "Pizza cam is live. Hawaiian is the first tray to go." },
];

export const rules = [
  "Bring a laptop. Build in the room. Demo from the room.",
  "Teams form in the portal. A star marks the leader.",
  "Presentations: volunteer your team. Three teams are picked from volunteers. One click per team.",
  "Submit a project from your team card before demos lock.",
  "Be decent. This is a meetup, not a product launch with a legal team.",
  "Cash prizes are organizer Interac e-Transfer deposit requests. Winners confirm a receiving email — Nightboard never asks for a bank login, SIN, or card.",
];

function m(name, role = "member") {
  return { name, role };
}

function t(id, name, members, extra = {}) {
  return {
    id,
    name,
    table: extra.table || "",
    status: extra.status || "building",
    oneLiner: extra.oneLiner || "Building tonight at ZayZoon.",
    need: extra.need || null,
    members,
    repo: extra.repo ?? null,
    mine: extra.mine || false,
    volunteered: extra.volunteered || false,
  };
}

/** Live portal roster from Cursor Calgary Meetup — August (35 teams). Floor x/y packing is owned elsewhere. */
export const teams = [
  t("jordan", "Jordan", [m("Jordan Defazio", "leader")]),
  t("techrage", "TechRage Inc.", [m("Jan Blasko"), m("Dmytro Prasolov", "leader")]),
  t("allen", "Allen", [m("Allen John", "leader")]),
  t("cursor-barely", "Cursor? I Barely Know Her", [m("Nicholas Irvine", "leader"), m("Amadeus")]),
  t("worklayer", "Worklayer", [m("Kush Gabani", "leader")]),
  t("nallay", "Nallay", [m("Muhammad Qanat Abbas", "leader"), m("Ujala Kiran"), m("Muhammad Ahmad Yar Khan")]),
  t("powercouple", "PowerCouple", [m("Katerina Kubisova"), m("Adam Petříček", "leader")]),
  t("gamonimuous", "gamonimuous", [m("Gauthier Appaix", "leader")]),
  t("pancake-shark", "Pancake Shark", [m("JJ", "leader")]),
  t("dissidia", "DISSIDIA", [m("Yupo Niu", "leader")]),
  t("blasian-duo", "Blasian Duo", [m("Sean Park"), m("Jonathan Elliott", "leader")]),
  t("spaceflex", "Spaceflex", [
    m("Sadhvi Sharma"),
    m("Daniel Collins", "leader"),
    m("Samuel Collins"),
    m("Nathan Nguyen"),
  ], { mine: true, volunteered: true, oneLiner: "Nightboard — live event OS for this room." }),
  t("figuring", "Figuring", [m("Shivam Khatri"), m("William Makino"), m("Abhijit Gowda", "leader")]),
  t("max-hum", "Max Hum", [m("Max Hum", "leader")]),
  t("lerman", "Lerman Mashynian", [m("Lerman Mashynian", "leader")]),
  t("promptforge", "PromptForge", [m("Jastegh Singh", "leader")]),
  t("two-birds", "2 Birds 1 Stone", [m("Masroor Syed"), m("Hamzah Umar", "leader")]),
  t("kohinoor", "kohinoorz awesome", [m("Kohinoor Chauhan", "leader")]),
  t("ethan", "Ethan", [m("Ethan Sam", "leader")]),
  t("barn", "Barn Softworks Inc.", [m("Haru", "leader")]),
  t("byron", "byron", [m("Byron Daniels", "leader")]),
  t("flash", "Flash", [m("Adarsh Menon", "leader")]),
  t("ninja-chips", "Ninja_chips", [m("Gordon", "leader"), m("Countless253")]),
  t("newbee", "NewBee", [m("Gourab Kishore Saha", "leader")]),
  t("transit-x", "Transit X", [m("Scott Li", "leader"), m("Yi Sheng"), m("Zhiping Zhang")]),
  t("zeroform", "ZeroForm", [m("Veselov Andriy", "leader")]),
  t("unnamed", "Unnamed", [m("Devry Lin", "leader")]),
  t("guilty", "Guilty as Charged", [m("Liam", "leader")]),
  t("gs-team", "GS-team", [m("Subramanian Narayanan", "leader"), m("Gaurav Dongol")]),
  t("randomguy1", "randomguy1", [m("Kundran", "leader")]),
  t("andrea", "Andrea", [m("Andrea Li", "leader")]),
  t("day-planner", "Day Planner", [m("Michael", "leader")]),
  t("pierre", "Pierre Tchio", [m("Pierre Tchio", "leader")]),
  t("rohit", "Rohit Romley", [m("Rohit Romley", "leader")]),
  t("kapre", "kapre", [m("Danish Ahmed", "leader")]),
];

export const demoQueue = [{ teamId: "spaceflex", minutes: 4 }];

/** People not on a team yet — join/offer targets. */
export const people = [
  {
    id: "priya",
    name: "Priya Shah",
    role: "Designer",
    skills: ["figma", "design", "copy"],
    lookingFor: "A team that needs UI tonight.",
  },
  {
    id: "omar",
    name: "Omar Haddad",
    role: "ML",
    skills: ["python", "ml", "eval"],
    lookingFor: "Open to joining a build.",
  },
];
