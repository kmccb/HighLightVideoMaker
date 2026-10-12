from pathlib import Path

from hlv import coach


def test_clean_reads_the_typed_list():
    got = coach.clean("Youngstown State, Coach Smith\n\n  Akron  \nMount Union,  Jones \nakron, Coach Dup\n")
    assert got == [{"school": "Youngstown State", "coach": "Coach Smith"}, {"school": "Akron", "coach": ""},
                   {"school": "Mount Union", "coach": "Jones"}]                # blank lines and a repeated school dropped
    assert coach.clean(got) == got                                            # and it reads back what it stored
    assert coach.clean(None) == [] and coach.clean("") == []
    assert len(coach.clean("\n".join(f"School {i}" for i in range(80)))) == coach.MAX


def test_file_names_keep_vertical_last_and_map_back():
    coaches = coach.clean("Youngstown State, Coach Smith\nMount Union")
    wide = coach.name_for(Path("reel/caleb_recruiting.mp4"), "Youngstown State")
    tall = coach.name_for(Path("reel/caleb_recruiting_vertical.mp4"), "Mount Union")
    assert wide == "caleb_recruiting_for_youngstown-state.mp4"
    assert tall == "caleb_recruiting_for_mount-union_vertical.mp4"
    assert coach.school_of(wide, coaches)["coach"] == "Coach Smith"
    assert coach.school_of(tall, coaches)["school"] == "Mount Union"
    assert coach.school_of("caleb_recruiting.mp4", coaches) is None


def test_card_is_drawn_at_both_shapes():
    pl = {"name": "Caleb Scott", "number": "12", "position": "WR / DB", "class_year": "2028", "school": "Poland Bulldogs",
          "color": "#1f4fd1", "accent": "#f5f4f0"}
    for size in ((1920, 1080), (1080, 1920)):
        img = coach.card(Path("."), pl, size, "A Very Long University Name Of Somewhere State", "Coach With A Long Name")
        assert img.shape == (size[1], size[0], 3) and img.max() > 200         # text was drawn, and it fits the frame
