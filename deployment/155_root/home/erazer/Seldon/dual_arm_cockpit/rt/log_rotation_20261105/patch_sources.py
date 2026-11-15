"""Exact-anchor logging-only changes to the verified installed source lineage."""
def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Source anchor changed: ' + old[:100])
    return source.replace(old, new, 1)


def patch_rt(source):
    source = '#include "rt_log_file.h"\n' + source
    source = once(source, 'FILE *fwrite_data = nullptr;',
                  'seldon::RtLogFile seldon_log;\nFILE *fwrite_data = nullptr;')
    source = once(source, 'if (queue_poptop(&save_item)){',
                  'if (queue_poptop(&save_item) && fwrite_data){')
    source = once(source, '            fprintf(fwrite_data, "\\n");',
                  '            fprintf(fwrite_data, "\\n");\n            fwrite_data = seldon_log.after_record();')
    start = source.index('void initialize_log_file(')
    end = source.index('\nint main(', start)
    source = source[:start] + '''void initialize_log_file(const std::string&) {
    seldon::require_log_volume();
    const auto directory = std::filesystem::path(seldon::rt_log_directory());
    std::filesystem::create_directories(directory);
    struct stat data{}, logs{};
    if (::stat("/mnt/data", &data) || ::stat(directory.c_str(), &logs) || data.st_dev != logs.st_dev)
        throw std::runtime_error("RT log path is outside the data volume");
    fwrite_data = seldon_log.open(directory);
}
''' + source[end:]
    source = once(source, '    if (fwrite_data) fclose(fwrite_data);',
                  '    seldon_log.close();\n    fwrite_data = nullptr;')
    return source


def patch_controller(source):
    source = once(source, 'import copy\n', 'import copy\nfrom .rt_logs import OutputCapture, log_path\n')
    source = once(source, "self.log = (self.root/'runtime/rt.log').open('ab', buffering=0)",
                  "self.log = OutputCapture(log_path('rt.log'))")
    start = source.index('        self.process = subprocess.Popen(')
    end = source.index('\n\n', start)
    block = source[start:end]
    source = source[:start] + '        try:\n' + '\n'.join('    ' + line for line in block.splitlines()) + '''
        except BaseException:
            self.log.close()
            self.log = None
            raise''' + source[end:]
    source = once(source, '''                        with (self.root/'runtime/rt.log').open('rb') as log:
                            log.seek(0, 2)
                            log.seek(max(0, log.tell()-3000))
                            self.rt_log = log.read().decode(errors='replace')''',
                  '''                        if self.log:
                            self.rt_log = self.log.tail()''')
    source = once(source, '        self.stop_journal.close()',
                  '        self.stop_journal.close()\n        if self.log:\n            self.log.close()')
    return source


def patch_startup(source):
    source = once(source, "BASE = 'http://127.0.0.1:8780'", "BASE = 'http://127.0.0.1:8780'\nRT_LOG = Path('/mnt/data/erazer/book_grasp/rt_logs/console/rt.log')")
    source = source.replace("str(ROOT/'runtime/rt.log')", 'str(RT_LOG)')
    source = once(source, "    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True)",
                  "    from seldon_arm.rt_logs import detached_output, log_path\n    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True)")
    source = once(source, "with (runtime/'start_arms.log').open('ab',buffering=0) as log:",
                  "with detached_output(log_path('backend.log')) as log:")
    source = source.replace("str(runtime/'start_arms.log')", "str(log_path('backend.log'))")
    return source
