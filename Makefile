CC = cc
PYTHON = python3
CFLAGS = -std=c11 -O3 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off
CPPFLAGS = -Iinclude -Isrc -Itools
BUILD = build
CORE = src/graph.c src/planner.c src/execute.c
OBJECTS = $(BUILD)/graph.o $(BUILD)/planner.o $(BUILD)/execute.o
HEADERS = include/tiny_arena.h src/internal.h

.PHONY: all test sanitize mutation bench publication-check clean FORCE
all: $(BUILD)/libtiny_arena.so $(BUILD)/benchmark $(BUILD)/test_native
$(BUILD):
	mkdir -p $@
FORCE:
$(BUILD)/%.o: src/%.c $(HEADERS) Makefile FORCE | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -c $< -o $@
$(BUILD)/workloads.o: tools/workloads.c tools/workloads.h include/tiny_arena.h Makefile FORCE | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -c $< -o $@
$(BUILD)/libtiny_arena.so: $(OBJECTS) $(BUILD)/workloads.o
	$(CC) -shared $^ -lm -o $@
$(BUILD)/benchmark: tools/benchmark.c $(OBJECTS) $(BUILD)/workloads.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -lm -o $@
$(BUILD)/test_native: tests/test_native.c $(OBJECTS) $(BUILD)/workloads.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -lm -o $@
test: all
	./$(BUILD)/test_native
	$(PYTHON) tests/test_planner.py $(BUILD)/libtiny_arena.so
	$(PYTHON) -O tests/test_planner.py $(BUILD)/libtiny_arena.so
	$(PYTHON) tests/check_no_heap.py $(OBJECTS)
	$(PYTHON) tests/test_reporting.py
	$(PYTHON) -O tests/test_reporting.py
sanitize: | $(BUILD)
	$(CC) $(CPPFLAGS) -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -fsanitize=address,undefined -fno-omit-frame-pointer tests/test_native.c $(CORE) tools/workloads.c -lm -o $(BUILD)/test_sanitize
	UBSAN_OPTIONS=halt_on_error=1 ./$(BUILD)/test_sanitize
mutation: | $(BUILD)
	$(PYTHON) tests/test_mutations.py
bench:
	$(PYTHON) tools/run_benchmark.py
publication-check:
	$(PYTHON) tools/check_publication.py
	$(PYTHON) tools/check_docs.py
	$(PYTHON) tools/verify_report.py results/published/host-20261014
clean:
	rm -rf $(BUILD)
