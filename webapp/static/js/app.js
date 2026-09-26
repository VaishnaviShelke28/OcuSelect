document.addEventListener("DOMContentLoaded", () => {

    const sidebar = document.getElementById("sidebar");
    const toggle = document.getElementById("sidebarToggle");

    if (sidebar && toggle) {

        toggle.addEventListener("click", () => {

            sidebar.classList.toggle("collapsed");

            document.body.classList.toggle(
                "sidebar-collapsed"
            );

            toggle.textContent =
                sidebar.classList.contains("collapsed")
                ? "›"
                : "‹";
        });
    }


    const mascot = document.querySelector(".hero-mascot");

    if (mascot) {

        mascot.addEventListener("click", () => {

            mascot.classList.remove("mascot-click");

            void mascot.offsetWidth;

            mascot.classList.add("mascot-click");
        });


        mascot.addEventListener("animationend", (event) => {

            if (event.animationName === "mascotHello") {
                mascot.classList.remove("mascot-click");
            }

        });
    }

});