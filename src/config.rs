pub const DEFAULT_PROMPT: &str = "O say can you see, by the dawn's early light,\nWhat so proudly we hail'd at the twilight's last gleaming,\nWhose broad stripes and bright stars through the perilous fight\nO'er the ramparts we watch'd were so gallantly streaming?\nAnd the rocket's red glare, the bomb bursting in air,\nGave proof through the night that our flag was still there,\nO say does that star-spangled banner yet wave\nO'er the land of the free and the home of the brave?\nOn the shore dimly seen through the mists of the deep\nWhere the foe's haughty host in dread silence reposes,\nWhat is that which the breeze, o'er the towering steep,\nAs it fitfully blows, half conceals, half discloses?\nNow it catches the gleam of the morning's first beam,\nIn full glory reflected now shines in the stream,\n'Tis the star-spangled banner - O long may it wave\nO'er the land of the free and the home of the brave!\nAnd where is that band who so vauntingly swore,\nThat the havoc of war and the battle's confusion\nA home and a Country should leave us no more?\nTheir blood has wash'd out their foul footstep's pollution.\nNo refuge could save the hireling and slave\nFrom the terror of flight or the gloom of the grave,\nAnd the star-spangled banner in triumph doth wave\nO'er the land of the free and the home of the brave.\nO thus be it ever when freemen shall stand\nBetween their lov'd home and the war's desolation!\nBlest with vict'ry and peace may the heav'n rescued land\nPraise the power that hath made and preserv'd us a nation!\nThen conquer we must, when our cause it is just,\nAnd this be our motto - \"In God is our trust,\"\nAnd the star-spangled banner in triumph shall wave\nO'er the land of the free and the home of the brave.";
pub const SYSTEM_MSG: &str = "Respond in JSON. Your task is to help us analyize bitcoin inscriptions. You will tell me a summary of the given text. And then give a list of up to 5 keywords that would describe the inscription.";

pub const SEED: u64 = 42;

pub const TEMPERATURE: f64 = 0.4;
pub const SAMPLE_LEN: usize = 1000;
pub const TOP_K: Option<usize> = None;
pub const TOP_P: Option<f64> = None;

pub const VERBOSE_PROMPT: bool = true;
pub const SPLIT_PROPMT: bool = true;
pub const REPEAT_PENALTY: f32 = 1.1;
pub const REPEAT_LAST_N: usize = 64;